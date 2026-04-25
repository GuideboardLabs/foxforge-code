from __future__ import annotations

import hashlib
import html
import json
import os
import re
import sqlite3
import time
from collections import deque
from html.parser import HTMLParser
import urllib.parse
import urllib.request
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from threading import Lock
from typing import Any

from shared_tools.db import connect as _db_connect_shared
from shared_tools.file_store import ProjectStore
from shared_tools.fact_policy import enrich_source_metadata, detect_topic_type, classify_fact_volatility
from shared_tools.domain_reputation import DomainReputation
from shared_tools.inference_router import InferenceRouter
from shared_tools.model_routing import lane_model_config, load_model_routing
from shared_tools.web_cache import WebQueryCache, cache_key as build_cache_key, normalize_query as normalize_cache_query, settings_digest as build_settings_digest


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class _InMemoryResponse:
    def __init__(self, payload: bytes) -> None:
        self._payload = payload
        self.status = 200
        self.headers: dict[str, Any] = {}

    def read(self) -> bytes:
        return bytes(self._payload)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        _ = exc_type
        _ = exc
        _ = tb
        return False


class _PageExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.title_parts: list[str] = []
        self.text_parts: list[str] = []
        self.links: list[str] = []
        self._in_title = False
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        key = tag.lower()
        if key in {"script", "style", "noscript", "svg", "nav", "header", "footer", "aside", "form", "iframe", "menu"}:
            self._skip_depth += 1
            return
        if key == "title":
            self._in_title = True
            return
        if key != "a":
            return
        for name, value in attrs:
            if name and name.lower() == "href" and value:
                self.links.append(value.strip())
                break

    def handle_endtag(self, tag: str) -> None:
        key = tag.lower()
        if key in {"script", "style", "noscript", "svg", "nav", "header", "footer", "aside", "form", "iframe", "menu"}:
            self._skip_depth = max(0, self._skip_depth - 1)
            return
        if key == "title":
            self._in_title = False

    def handle_data(self, data: str) -> None:
        if self._skip_depth > 0:
            return
        text = " ".join(str(data or "").split())
        if not text:
            return
        if self._in_title:
            self.title_parts.append(text)
        self.text_parts.append(text)

    def title(self) -> str:
        return " ".join(self.title_parts).strip()

    def snippet(self, max_chars: int = 600) -> str:
        text = " ".join(self.text_parts).strip()
        if len(text) <= max_chars:
            return text
        cut = text[:max_chars].rsplit(" ", 1)[0].strip()
        return (cut or text[:max_chars]).strip() + "..."


_MD_LINK_ONLY_RE = re.compile(r"^\s*(\[([^\]]*)\]\([^)]*\)\s*[|•·\-,]?\s*)+\s*$")
_BOILERPLATE_LOWER = (
    "subscribe to our newsletter",
    "sign up for our newsletter",
    "this site uses cookies",
    "we use cookies",
    "accept all cookies",
    "cookie preferences",
    "manage cookies",
    "skip to main content",
    "skip to content",
    "all rights reserved",
    "share on twitter",
    "share on facebook",
    "share on linkedin",
)


_NAV_HEADING_RE = re.compile(
    r"^#+\s*(main navigation|navigation|site navigation|primary navigation|header|footer|menu|breadcrumb|skip to|table of contents|contents)\s*$",
    re.IGNORECASE,
)
_JUNK_SECTION_HEADING_RE = re.compile(
    r"^#+\s*(related articles?|you might also like|recommended for you|more stories|trending now|most popular|also read|see also|more from|popular posts?|suggested reading)\s*$",
    re.IGNORECASE,
)
_AD_LINE_RE = re.compile(
    r"\b(sponsored|advertisement|promoted content|partner content|paid content|ad\b)",
    re.IGNORECASE,
)
_BOILERPLATE_LOWER_EXTENDED = _BOILERPLATE_LOWER + (
    "skip to main content",
    "skip to content",
    "skip to navigation",
    "back to top",
    "jump to navigation",
    "view source",
    "edit this page",
    "log in",
    "create account",
    "privacy policy",
    "terms of use",
    "contact us",
    "about us",
    "advertise with us",
    "subscribe now",
    "sign in to comment",
    "enable javascript",
    "please enable",
    "get the app",
    "download the app",
    "follow us on",
    "newsletter signup",
    "you might also like",
    "related articles",
    "recommended for you",
    "sponsored content",
    "advertisement",
    "promoted",
    "read more at",
    "click here to",
    "learn more about",
    "trending now",
    "most popular",
    "don't miss",
    "limited time",
    "free shipping",
    "add to cart",
    "buy now",
    "shop now",
    "sale ends",
    "discount code",
    "partner content",
    "paid content",
    "affiliate",
)


def _clean_crawl4ai_markdown(text: str) -> str:
    """Strip navigation link menus, cookie banners, ad content, and share-button boilerplate from Crawl4AI markdown."""
    if not text:
        return text
    lines = text.split("\n")
    cleaned: list[str] = []
    nav_run = 0
    skip_nav_section = False
    skip_junk_section = False
    for line in lines:
        stripped = line.strip()

        # Detect a navigation heading and skip until next content heading
        if _NAV_HEADING_RE.match(stripped):
            skip_nav_section = True
            skip_junk_section = False
            continue
        if skip_nav_section:
            if re.match(r"^#+\s+\S", stripped) and not _NAV_HEADING_RE.match(stripped):
                skip_nav_section = False
                # fall through to process this heading normally
            else:
                continue

        # Detect junk section headings (related articles, you might also like, etc.)
        if _JUNK_SECTION_HEADING_RE.match(stripped):
            skip_junk_section = True
            continue
        if skip_junk_section:
            if re.match(r"^#+\s+\S", stripped) and not _JUNK_SECTION_HEADING_RE.match(stripped):
                skip_junk_section = False
                # fall through to process this heading normally
            else:
                continue

        # Lines composed entirely of markdown links — navigation menus, breadcrumbs
        if stripped and _MD_LINK_ONLY_RE.match(stripped):
            nav_run += 1
            if nav_run <= 1:
                cleaned.append(line)  # keep first link (may be a real article ref)
            continue
        else:
            nav_run = 0

        # Short boilerplate phrases
        low = stripped.lower()
        if len(stripped) < 160 and any(p in low for p in _BOILERPLATE_LOWER_EXTENDED):
            continue

        # Short ad/sponsored lines (<200 chars)
        if len(stripped) < 200 and _AD_LINE_RE.search(stripped):
            continue

        # Lines where >75% of non-whitespace chars are part of markdown link syntax
        if stripped and len(stripped) < 300:
            link_chars = sum(len(m.group(0)) for m in re.finditer(r"\[([^\]]*)\]\([^)]*\)", stripped))
            if link_chars > 0 and link_chars / len(stripped) > 0.75:
                continue

        # Lone image-only lines (![...](...)) with no surrounding text
        if re.match(r"^!\[[^\]]*\]\([^)]*\)\s*$", stripped):
            continue

        cleaned.append(line)

    # Collapse 3+ blank lines → 2
    result: list[str] = []
    blanks = 0
    for line in cleaned:
        if not line.strip():
            blanks += 1
            if blanks <= 2:
                result.append(line)
        else:
            blanks = 0
            result.append(line)
    return "\n".join(result)


def build_web_progress_payload(result: dict[str, Any] | None) -> dict[str, Any]:
    """Normalize a web-run result into a compact progress payload for the UI."""
    details = result if isinstance(result, dict) else {}
    tier_counts = {}
    scoring = details.get("source_scoring_summary")
    if isinstance(scoring, dict):
        maybe_tiers = scoring.get("tier_counts")
        if isinstance(maybe_tiers, dict):
            tier_counts = maybe_tiers
    raw_sources = details.get("sources") or []
    used_sources: list[dict[str, Any]] = []
    for src in raw_sources:
        if not isinstance(src, dict):
            continue
        url = str(src.get("url") or src.get("source_url") or "").strip()
        domain = str(src.get("source_domain") or src.get("domain") or "").strip().lower()
        if not domain and url:
            try:
                from urllib.parse import urlparse
                domain = urlparse(url).hostname or ""
                domain = domain.removeprefix("www.")
            except Exception:
                pass
        if not domain and not url:
            continue
        used_sources.append(
            {
                "domain": domain,
                "url": url or (f"https://{domain}" if domain else ""),
                "title": str(src.get("title") or "").strip(),
                "tier": str(src.get("source_tier") or src.get("tier") or "").strip(),
                "score": float(src.get("source_score", 0.0) or 0.0),
            }
        )
    return {
        "note": "Web stack ready.",
        "mode": str(details.get("mode", "")),
        "source_count": int(details.get("source_count", 0) or 0),
        "seed_count": int(details.get("seed_count", 0) or 0),
        "crawl_pages": int(details.get("crawl_pages", 0) or 0),
        "crawl_gated_links": int(details.get("crawl_gated_links", 0) or 0),
        "query_variants_count": int(details.get("query_variants_count", 0) or 0),
        "conflict_count": int(details.get("conflict_count", 0) or 0),
        "tier1": int(tier_counts.get("tier1", 0) or 0),
        "tier2": int(tier_counts.get("tier2", 0) or 0),
        "tier3": int(tier_counts.get("tier3", 0) or 0),
        "banned_sources": int(tier_counts.get("banned_sources", 0) or 0),
        "web_sources": used_sources,
    }


class WebResearchEngine:
    VALID_MODES = {"off", "ask", "auto"}
    VALID_PROVIDERS = {"auto", "searxng", "duckduckgo_html", "duckduckgo_api"}
    # General developer-focused source tiers.
    # Tier 1: official/primary sources for coding and development: language docs,
    # standards bodies, package registries, platform docs, release/security sources,
    # and project-owned documentation. These should dominate for implementation
    # details, compatibility, API behavior, vulnerability status, and setup steps.
    TRUST_TIER_1 = {
        # Standards, specifications, and interoperability references.
        "w3.org",
        "whatwg.org",
        "tc39.es",
        "ecma-international.org",
        "ietf.org",
        "rfc-editor.org",
        "iana.org",
        "unicode.org",
        "opengroup.org",
        "openapis.org",
        "json-schema.org",
        "semver.org",
        "spdx.org",
        "cyclonedx.org",
        "slsa.dev",
        "sigstore.dev",
        "opencontainers.org",
        "cncf.io",
        "linuxfoundation.org",

        # Security, vulnerability, and supply-chain primary sources.
        "cisa.gov",
        "nist.gov",
        "nvd.nist.gov",
        "cve.org",
        "mitre.org",
        "owasp.org",
        "osv.dev",
        "deps.dev",
        "rustsec.org",
        "first.org",
        "cert.org",
        "cert.europa.eu",
        "advisories.gitlab.com",
        "security.snyk.io",
        "portswigger.net",

        # Source-control and developer platform documentation.
        "docs.github.com",
        "docs.gitlab.com",
        "developer.atlassian.com",
        "support.atlassian.com",
        "git-scm.com",
        "gitforwindows.org",
        "mercurial-scm.org",
        "subversion.apache.org",

        # Language, runtime, and standard-library documentation.
        "developer.mozilla.org",
        "docs.python.org",
        "python.org",
        "peps.python.org",
        "nodejs.org",
        "deno.com",
        "bun.sh",
        "typescriptlang.org",
        "go.dev",
        "pkg.go.dev",
        "rust-lang.org",
        "doc.rust-lang.org",
        "rustup.rs",
        "ruby-lang.org",
        "ruby-doc.org",
        "php.net",
        "perl.org",
        "perldoc.perl.org",
        "docs.oracle.com",
        "openjdk.org",
        "kotlinlang.org",
        "scala-lang.org",
        "docs.swift.org",
        "swift.org",
        "dart.dev",
        "elixir-lang.org",
        "hexdocs.pm",
        "erlang.org",
        "haskell.org",
        "clojure.org",
        "racket-lang.org",
        "lua.org",
        "gnu.org",
        "gcc.gnu.org",
        "llvm.org",
        "clang.llvm.org",
        "ziglang.org",
        "nim-lang.org",
        "julia-lang.org",
        "r-project.org",
        "cran.r-project.org",

        # Official package registries and package manager docs.
        "pypi.org",
        "npmjs.com",
        "docs.npmjs.com",
        "crates.io",
        "docs.rs",
        "rubygems.org",
        "packagist.org",
        "getcomposer.org",
        "nuget.org",
        "search.maven.org",
        "central.sonatype.com",
        "repo.maven.apache.org",
        "maven.apache.org",
        "gradle.org",
        "plugins.gradle.org",
        "pub.dev",
        "hex.pm",
        "clojars.org",
        "conda.io",
        "anaconda.org",
        "brew.sh",
        "vcpkg.io",
        "conan.io",
        "packages.debian.org",
        "packages.ubuntu.com",
        "archlinux.org",
        "alpinelinux.org",
        "fedoraproject.org",
        "chocolatey.org",
        "scoop.sh",

        # Web, frontend, testing, and JavaScript ecosystem docs.
        "web.dev",
        "developer.chrome.com",
        "webkit.org",
        "react.dev",
        "nextjs.org",
        "vuejs.org",
        "angular.dev",
        "angular.io",
        "svelte.dev",
        "kit.svelte.dev",
        "solidjs.com",
        "preactjs.com",
        "astro.build",
        "nuxt.com",
        "remix.run",
        "emberjs.com",
        "jquery.com",
        "vitejs.dev",
        "vitest.dev",
        "webpack.js.org",
        "rollupjs.org",
        "parceljs.org",
        "babeljs.io",
        "eslint.org",
        "prettier.io",
        "pnpm.io",
        "yarnpkg.com",
        "jestjs.io",
        "playwright.dev",
        "cypress.io",
        "testing-library.com",
        "storybook.js.org",
        "tailwindcss.com",
        "getbootstrap.com",
        "mui.com",
        "chakra-ui.com",
        "sass-lang.com",
        "postcss.org",
        "docusaurus.io",
        "vitepress.dev",

        # Python data, backend, and application framework docs.
        "django-project.com",
        "flask.palletsprojects.com",
        "palletsprojects.com",
        "fastapi.tiangolo.com",
        "pydantic.dev",
        "sqlalchemy.org",
        "alembic.sqlalchemy.org",
        "numpy.org",
        "pandas.pydata.org",
        "scipy.org",
        "scikit-learn.org",
        "matplotlib.org",
        "seaborn.pydata.org",
        "plotly.com",
        "bokeh.org",
        "streamlit.io",
        "docs.pytest.org",
        "pytest.org",
        "sphinx-doc.org",
        "mkdocs.org",
        "jupyter.org",
        "ipython.org",
        "celeryq.dev",
        "scrapy.org",

        # Java, JVM, .NET, mobile, and application platform docs.
        "spring.io",
        "quarkus.io",
        "micronaut.io",
        "jakarta.ee",
        "hibernate.org",
        "junit.org",
        "jetbrains.com",
        "learn.microsoft.com",
        "docs.microsoft.com",
        "dotnet.microsoft.com",
        "developer.android.com",
        "firebase.google.com",
        "developer.apple.com",
        "docs.flutter.dev",
        "flutter.dev",
        "reactnative.dev",
        "expo.dev",

        # Cloud, infrastructure, orchestration, and observability docs.
        "docs.aws.amazon.com",
        "aws.amazon.com",
        "cloud.google.com",
        "developers.google.com",
        "ai.google.dev",
        "cloudflare.com",
        "developers.cloudflare.com",
        "azure.microsoft.com",
        "ibm.com",
        "docs.oracle.com",
        "docs.digitalocean.com",
        "kubernetes.io",
        "helm.sh",
        "docker.com",
        "docs.docker.com",
        "podman.io",
        "containerd.io",
        "terraform.io",
        "developer.hashicorp.com",
        "ansible.com",
        "docs.ansible.com",
        "pulumi.com",
        "prometheus.io",
        "grafana.com",
        "opentelemetry.io",
        "nginx.org",
        "envoyproxy.io",
        "istio.io",
        "linkerd.io",
        "consul.io",
        "nomadproject.io",

        # Databases, search, queues, and data infrastructure docs.
        "postgresql.org",
        "sqlite.org",
        "mysql.com",
        "dev.mysql.com",
        "mariadb.com",
        "redis.io",
        "mongodb.com",
        "elastic.co",
        "opensearch.org",
        "clickhouse.com",
        "duckdb.org",
        "cassandra.apache.org",
        "couchbase.com",
        "neo4j.com",
        "influxdata.com",
        "cockroachlabs.com",
        "supabase.com",
        "planetscale.com",
        "prisma.io",
        "graphql.org",
        "grpc.io",
        "protobuf.dev",
        "rabbitmq.com",
        "kafka.apache.org",

        # ML, AI, data-science, and numerical-computing docs.
        "pytorch.org",
        "tensorflow.org",
        "keras.io",
        "jax.readthedocs.io",
        "huggingface.co",
        "docs.openai.com",
        "platform.openai.com",
        "docs.anthropic.com",
        "anthropic.com",
        "onnx.ai",
        "mlflow.org",
        "xgboost.readthedocs.io",
        "lightgbm.readthedocs.io",
        "catboost.ai",
        "ray.io",
        "dask.org",
        "numba.pydata.org",

        # Operating-system, manuals, and foundational project docs.
        "kernel.org",
        "docs.kernel.org",
        "man7.org",
        "freebsd.org",
        "debian.org",
        "ubuntu.com",
        "wiki.archlinux.org",
        "gentoo.org",
        "opensuse.org",
        "systemd.io",
        "freedesktop.org",
        "x.org",
        "wayland.freedesktop.org",
        "apache.org",
        "eclipse.org",
        "mozilla.org",
    }

    # Tier 2: strong secondary sources for developer work: maintained repos,
    # Q&A communities, reputable technical publishers, vendor engineering blogs,
    # and high-quality tutorials. Useful for examples, edge cases, and tradeoffs,
    # but they should not outrank primary docs for exact API behavior.
    TRUST_TIER_2 = {
        # Source hosting, issue trackers, code review, and project hubs.
        "github.com",
        "gitlab.com",
        "bitbucket.org",
        "codeberg.org",
        "sourceforge.net",
        "launchpad.net",
        "savannah.gnu.org",
        "gerrit-review.googlesource.com",
        "go-review.googlesource.com",
        "chromium.googlesource.com",
        "android.googlesource.com",

        # Developer Q&A and focused technical communities.
        "stackoverflow.com",
        "stackexchange.com",
        "serverfault.com",
        "superuser.com",
        "askubuntu.com",
        "dba.stackexchange.com",
        "security.stackexchange.com",
        "unix.stackexchange.com",
        "softwareengineering.stackexchange.com",
        "codereview.stackexchange.com",
        "math.stackexchange.com",
        "crossvalidated.com",
        "discuss.python.org",
        "discuss.rubyonrails.org",
        "users.rust-lang.org",
        "internals.rust-lang.org",
        "discuss.kotlinlang.org",
        "forum.djangoproject.com",
        "discuss.pytorch.org",
        "discuss.tensorflow.org",
        "forums.docker.com",
        "discuss.kubernetes.io",
        "discourse.llvm.org",

        # Documentation hosting and API/doc aggregators.
        "readthedocs.io",
        "readthedocs.org",
        "devdocs.io",
        "pkgdown.r-lib.org",
        "docs.readme.com",
        "stoplight.io",
        "redocly.com",
        "swagger.io",
        "postman.com",
        "insomnia.rest",

        # Reputable technical education and implementation guides.
        "realpython.com",
        "fullstackpython.com",
        "testdriven.io",
        "pythonspeed.com",
        "python-poetry.org",
        "realworldpython.com",
        "baeldung.com",
        "mkyong.com",
        "reflectoring.io",
        "vladmihalcea.com",
        "callicoder.com",
        "css-tricks.com",
        "smashingmagazine.com",
        "webplatform.news",
        "2ality.com",
        "surma.dev",
        "overreacted.io",
        "kentcdodds.com",
        "joshwcomeau.com",
        "robinwieruch.de",
        "digitalocean.com",
        "freecodecamp.org",
        "learnxinyminutes.com",
        "refactoring.guru",
        "martinfowler.com",
        "patterns.dev",
        "roadmap.sh",
        "thoughtworks.com",
        "atlassian.com",
        "stripe.com",
        "twilio.com",
        "shopify.dev",
        "vercel.com",
        "netlify.com",
        "heroku.com",
        "fly.io",
        "render.com",
        "railway.app",
        "auth0.com",
        "okta.com",

        # Engineering blogs and vendor technical blogs.
        "engineering.fb.com",
        "netflixtechblog.com",
        "engineering.linkedin.com",
        "engineering.atspotify.com",
        "dropbox.tech",
        "blog.cloudflare.com",
        "devblogs.microsoft.com",
        "github.blog",
        "about.gitlab.com",
        "blog.rust-lang.org",
        "v8.dev",
        "chromium.org",
        "engineering.salesforce.com",
        "engineering.shopify.com",
        "slack.engineering",
        "eng.uber.com",
        "uber.com",
        "lyft.com",
        "airbnb.io",
        "medium.engineering",
        "engineering.atmeta.com",

        # Security research, incident analysis, and practitioner publications.
        "krebsonsecurity.com",
        "bleepingcomputer.com",
        "darkreading.com",
        "securityweek.com",
        "threatpost.com",
        "talosintelligence.com",
        "unit42.paloaltonetworks.com",
        "projectzero.google",
        "googleprojectzero.blogspot.com",
        "blog.trailofbits.com",
        "trailofbits.com",
        "blog.qualys.com",
        "rapid7.com",
        "veracode.com",
        "snyk.io",
        "socket.dev",
        "wiz.io",
        "semgrep.dev",

        # Technical news and systems/Linux publications.
        "lwn.net",
        "phoronix.com",
        "thenewstack.io",
        "infoq.com",
        "arstechnica.com",
        "theregister.com",
        "infoworld.com",
        "zdnet.com",
        "techtarget.com",
        "itnext.io",
        "towardsdatascience.com",
        "lilianweng.github.io",
        "sebastianraschka.com",
        "distill.pub",
        "paperswithcode.com",
        "arxiv.org",
        "semanticscholar.org",

        # Practical operations, Linux, and hosting guides.
        "redhat.com",
        "canonical.com",
        "linuxize.com",
        "linuxhandbook.com",
        "linuxjournal.com",
        "howtoforge.com",
        "phoenixnap.com",
        "linode.com",
        "akamai.com",
        "cloudsmith.com",
        "jfrog.com",
        "sonatype.com",
    }

    # Tier 3: usable but variable developer sources. These are often community,
    # SEO-heavy, beginner-oriented, paywalled, anecdotal, or weakly maintained.
    # Keep them as fallback/context only; do not prefer them over tiers 1-2.
    TRUST_TIER_3 = {
        # Social/community feeds and anecdotal discussion.
        "reddit.com",
        "old.reddit.com",
        "news.ycombinator.com",
        "lobste.rs",
        "dev.to",
        "medium.com",
        "hashnode.com",
        "substack.com",
        "linkedin.com",
        "quora.com",
        "producthunt.com",
        "mastodon.social",
        "fosstodon.org",
        "lemmy.world",

        # Broad tutorial and training sites with mixed technical depth.
        "w3schools.com",
        "geeksforgeeks.org",
        "tutorialspoint.com",
        "javatpoint.com",
        "programiz.com",
        "guru99.com",
        "educative.io",
        "codecademy.com",
        "datacamp.com",
        "pluralsight.com",
        "coursera.org",
        "udemy.com",
        "udacity.com",
        "simplilearn.com",
        "edureka.co",
        "intellipaat.com",
        "hackr.io",
        "sololearn.com",
        "codeacademy.com",
        "codedamn.com",
        "scrimba.com",

        # Mixed-quality dev blogs, snippet sites, and magazine/community posts.
        "hackernoon.com",
        "dzone.com",
        "sitepoint.com",
        "codeproject.com",
        "code-maze.com",
        "toptal.com",
        "bitsrc.io",
        "blog.logrocket.com",
        "blog.bitsrc.io",
        "daily.dev",
        "earthly.dev",
        "betterprogramming.pub",
        "plainenglish.io",
        "towardsdev.com",
        "analyticsvidhya.com",
        "neptune.ai",
        "machinelearningmastery.com",
        "kdnuggets.com",
        "paperspace.com",
        "assemblyai.com",
        "openreplay.com",

        # Admin, hosting, and Linux how-to content that can be useful but varies.
        "linuxhint.com",
        "tecmint.com",
        "itsfoss.com",
        "ostechnix.com",
        "tecadmin.net",
        "serverlab.ca",
        "hostinger.com",
        "kinsta.com",
        "wpbeginner.com",
        "ionos.com",
        "cloudways.com",
        "namecheap.com",
        "godaddy.com",
        "bluehost.com",

        # Secondary mirrors, examples, and reference aggregators.
        "mankier.com",
        "linux.die.net",
        "ss64.com",
        "commandlinux.com",
        "explainshell.com",
        "devhints.io",
        "cheatography.com",
        "cheatsheetseries.owasp.org",
        "quickref.me",
        "runebook.dev",
        "devdoc.net",
        "docs4dev.com",
        "rosettacode.org",
        "example-code.com",
        "zetcode.com",
        "w3resource.com",
        "includehelp.com",
    }

    # Banned sources: domains that are usually poor inputs for code/development
    # web-crawl context. They are blocked by scoring (quality_blocked=True) rather
    # than merely demoted, because they are commonly social shells, shorteners,
    # scraped copies, spam, paste dumps, job/review marketplaces, or CDN/file hosts
    # with little explanatory content.
    BANNED_SOURCES = {
        # Social/media shells and video/image pages with low crawl text signal.
        "facebook.com",
        "m.facebook.com",
        "instagram.com",
        "tiktok.com",
        "threads.net",
        "snapchat.com",
        "pinterest.com",
        "x.com",
        "twitter.com",
        "t.co",
        "youtube.com",
        "youtu.be",
        "vimeo.com",
        "dailymotion.com",
        "twitch.tv",
        "imgur.com",
        "flickr.com",
        "giphy.com",
        "tumblr.com",
        "vk.com",
        "weibo.com",

        # URL shorteners and tracking redirectors.
        "bit.ly",
        "tinyurl.com",
        "goo.gl",
        "ow.ly",
        "buff.ly",
        "rebrand.ly",
        "cutt.ly",
        "is.gd",
        "s.id",
        "lnkd.in",
        "ift.tt",
        "trib.al",
        "tiny.cc",
        "shorturl.at",
        "rb.gy",

        # Paste dumps, raw snippets, and anonymous file-drop pages.
        "pastebin.com",
        "hastebin.com",
        "paste.rs",
        "ghostbin.co",
        "rentry.co",
        "justpaste.it",
        "controlc.com",
        "codepad.org",
        "ideone.com",
        "replit.com",
        "codesandbox.io",
        "jsfiddle.net",
        "codepen.io",

        # Stack Overflow/Q&A scraper or answer-farm copies.
        "codegrepper.com",
        "devasking.com",
        "stackoom.com",
        "copyprogramming.com",
        "programmersought.com",
        "newbedev.com",
        "pythonfixing.com",
        "errorsfixing.com",
        "anycodings.com",
        "syntaxfix.com",
        "tutorialguruji.com",
        "stackovercoder.com",
        "stackoverrun.com",
        "qastack.com",
        "qastack.jp",
        "qastack.cn",
        "qastack.mx",
        "qastack.id",
        "qastack.kr",
        "itecnote.com",
        "iditect.com",
        "helpex.vn",
        "issueantenna.com",
        "bleepcoder.com",
        "gitmemory.com",
        "lightrun.com",

        # CDN/file mirrors and generated package file hosts without explanation.
        "unpkg.com",
        "cdn.jsdelivr.net",
        "cdnjs.cloudflare.com",
        "rawgit.com",
        "raw.githack.com",
        "raw.githubusercontent.com",
        "gist.githubusercontent.com",
        "gitcdn.xyz",
        "bundlephobia.com",

        # Marketplaces, company-review sites, and mostly non-technical listings.
        "g2.com",
        "capterra.com",
        "trustpilot.com",
        "glassdoor.com",
        "indeed.com",
        "monster.com",
        "ziprecruiter.com",
        "upwork.com",
        "fiverr.com",
        "freelancer.com",
        "peopleperhour.com",
        "appsumo.com",
        "alternativeto.net",
        "slant.co",
        "saashub.com",

        # Generic document/social sharing and consumer wiki/how-to pages.
        "scribd.com",
        "slideshare.net",
        "academia.edu",
        "researchgate.net",
        "prezi.com",
        "issuu.com",
        "wikihow.com",
        "ehow.com",
        "answers.com",
        "answers.microsoft.com",
        "support.google.com",
    }

    PROPAGANDA_TERMS = {
        "shocking", "you won't believe", "exposed", "bombshell", "destroyed",
        "humiliated", "secret agenda", "mainstream media won't", "cover-up",
        "they don't want you to know", "leaked", "breaking truth",
    }
    LOW_SIGNAL_TEXT_TERMS = {
        "unsupported client",
        "unsupported browser",
        "please update your browser",
        "enable javascript",
        "please enable javascript",
        "enable cookies",
        "verify you are human",
        "checking your browser",
        "access denied",
        "request blocked",
        "forbidden",
        "service unavailable",
        "temporarily unavailable",
        "captcha",
        "are you a robot",
        "cloudflare",
        "security check",
        "browser is not supported",
    }
    LOW_SIGNAL_URL_TERMS = {
        "/cdn-cgi/",
        "/captcha",
        "/challenge",
        "/unsupported",
        "/error",
        "/forbidden",
        "/access-denied",
        "/blocked",
        "cf_chl",
        "cf-chl",
    }
    NAVIGATION_NOISE_TERMS = {
        "skip to content",
        "book a demo",
        "privacy policy",
        "terms of service",
        "all rights reserved",
        "cookie preferences",
        "log in",
        "sign up",
        "create account",
    }

    TOPIC_FAMILY_ALIASES = {
        "pet_care": "animal_care",
        "pets": "animal_care",
        "pet_health": "animal_care",
        "veterinary": "animal_care",
        "vet": "animal_care",
    }
    TOPIC_HINTS = {
        "technical": (
            "official documentation",
            "release notes",
            "version compatibility",
            "changelog",
        ),
        "finance": (
            "sec filing",
            "earnings release",
            "guidance update",
            "analyst consensus",
        ),
        "current_events": (
            "official statement",
            "timeline",
            "live updates",
            "breaking news",
        ),
        "law": (
            "statute text",
            "court ruling",
            "effective date",
            "official guidance",
        ),
        "education": (
            "admissions deadline",
            "accreditation status",
            "official program page",
            "curriculum requirements",
        ),
        "travel": (
            "entry requirements",
            "visa rules",
            "travel advisory",
            "official guidance",
        ),
        "animal_care": (
            "veterinary guidance",
            "species specific recommendations",
            "animal welfare guidance",
            "official guidance",
        ),
        "food": (
            "nutrition facts",
            "ingredient list",
            "food safety guidance",
            "official guidance",
        ),
        "books": (
            "publication date",
            "edition details",
            "publisher announcement",
            "author interview",
        ),
        "parenting": (
            "pediatric guideline",
            "age recommendation",
            "development milestone",
            "safety guidance",
        ),
        "business": (
            "quarterly results",
            "management guidance",
            "industry outlook",
            "official filing",
        ),
        "real_estate": (
            "mortgage rates",
            "housing inventory",
            "median home price",
            "official market report",
        ),
        "gaming": (
            "patch notes",
            "release date",
            "developer update",
            "season roadmap",
        ),
        "automotive": (
            "msrp",
            "recall notice",
            "range mpg",
            "official spec sheet",
        ),
        "tv_shows": (
            "season release date",
            "episode schedule",
            "official network announcement",
            "renewed cancelled",
        ),
        "movies": (
            "release date",
            "box office",
            "official trailer",
            "festival premiere",
        ),
        "music": (
            "album release date",
            "tour dates",
            "official announcement",
            "chart update",
        ),
        "art": (
            "exhibition dates",
            "museum announcement",
            "auction result",
            "artist statement",
        ),
        "sports": (
            "schedule",
            "results",
            "standings",
            "roster",
            "official announcement",
        ),
        "combat_sports": (
            "fight card",
            "bout order",
            "main event",
            "official announcement",
        ),
        "sports_event": (
            "fight card",
            "results",
            "official card",
            "main event",
        ),
    }

    # Topics for which a full Wikipedia article is always fetched as a primary source.
    # The chat layer prefers quick Wikipedia grounding whenever it can answer safely.
    WIKIPEDIA_TOPIC_TYPES: frozenset[str] = frozenset({
        "history",
        "technical",
        "science",
        "books",
        "art",
        "education",
        "travel",
        "food",
        "business",
        "politics",
        "sports",
        "combat_sports",
        "sports_event",
        "current_events",
        "movies",
        "tv_shows",
        "music",
        "gaming",
        "general",
    })

    def __init__(self, repo_root: Path) -> None:
        self.repo_root = repo_root
        self.store = ProjectStore(repo_root)
        self.root = repo_root / "Runtime" / "web"
        self.pending_path = self.root / "pending_requests.json"
        self.settings_path = self.root / "settings.json"
        self.sources_log_path = self.root / "sources.jsonl"
        self.lock = Lock()
        self._searxng_backoff_until = 0.0
        self._crawl4ai_backoff_until = 0.0
        self._reddit_backoff_until = 0.0
        self._tor_active = False  # Set True during underground run_query execution

        self.root.mkdir(parents=True, exist_ok=True)
        self._domain_rep = DomainReputation(repo_root)
        self._query_cache = WebQueryCache(repo_root)
        if not self.pending_path.exists():
            self.pending_path.write_text("[]", encoding="utf-8")
        if not self.settings_path.exists():
            self.settings_path.write_text(
                json.dumps(
                    {
                        "mode": "auto",
                        "provider": "auto",
                        "max_results": 8,
                        "query_expansion_enabled": True,
                        "query_expansion_variants": 4,
                        "query_decomposition_enabled": True,
                        "query_decomposition_max_sub": 5,
                        "pre_crawl_seed_selection_enabled": True,
                        "pre_crawl_results_per_query": 20,
                        "pre_crawl_primary_quota": 5,
                        "pre_crawl_extra_quota_min": 2,
                        "pre_crawl_extra_quota_max": 3,
                        "smart_query_variants_enabled": False,
                        "smart_query_variants_limit": 3,
                        "smart_query_summary_chars": 2200,
                        "smart_query_cache_rows": 6,
                        "iterative_search_enabled": True,
                        "iterative_search_time_budget_sec": 25,
                        "embedding_content_filter_enabled": True,
                        "source_scoring_enabled": True,
                        "min_quality_sources": 2,
                        "context_min_source_score": 0.62,
                        "fresh_runs_enabled": True,
                        "fresh_runs_history_limit": 6,
                        "fresh_runs_min_new_domains": 4,
                        "conflict_detection_enabled": True,
                        "crawl_relevance_gating_enabled": False,
                        "crawl_relevance_min_score": 0.1,
                        "searxng_base_url": "http://127.0.0.1:8080",
                        "searxng_timeout_sec": 20,
                        "searxng_engines": "",
                        "searxng_categories": "",
                        "searxng_language": "",
                        "crawl_enabled": True,
                        "crawl_depth": 2,
                        "crawl_max_pages": 18,
                        "crawl_links_per_page": 8,
                        "crawl_timeout_sec": 0,
                        "crawl4ai_enabled": True,
                        "crawl4ai_base_url": "http://127.0.0.1:11235",
                        "crawl4ai_timeout_sec": 40,
                        "crawl4ai_retry_attempts": 2,
                        "crawl4ai_css_selector": "",
                        "newspaper_enabled": True,
                        "newspaper_language": "",
                        "search_retry_attempts": 3,
                        "crawl_retry_attempts": 3,
                        "crawl_same_domain_only": True,
                        "crawl_text_chars": 1200,
                    },
                    indent=2,
                    ensure_ascii=True,
                ),
                encoding="utf-8",
            )
        if not self.sources_log_path.exists():
            self.sources_log_path.write_text("", encoding="utf-8")

    def _load_pending(self) -> list[dict[str, Any]]:
        try:
            data = json.loads(self.pending_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return []
        if isinstance(data, list):
            return [x for x in data if isinstance(x, dict)]
        return []

    def _save_pending(self, rows: list[dict[str, Any]]) -> None:
        self.pending_path.write_text(json.dumps(rows, indent=2, ensure_ascii=True), encoding="utf-8")

    def _load_settings(self) -> dict[str, Any]:
        try:
            data = json.loads(self.settings_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            data = {}
        if not isinstance(data, dict):
            data = {}

        def _coerce_bool(value: Any, *, default: bool) -> bool:
            if value is None:
                return default
            if isinstance(value, bool):
                return value
            if isinstance(value, (int, float)):
                return bool(value)
            text = str(value).strip().lower()
            if text in {"1", "true", "yes", "on"}:
                return True
            if text in {"0", "false", "no", "off"}:
                return False
            return default

        def _bool_setting(name: str, default: bool) -> bool:
            return _coerce_bool(data.get(name, default), default=default)

        mode = str(data.get("mode", "auto")).strip().lower()
        if mode not in self.VALID_MODES:
            mode = "auto"
        data["mode"] = mode
        provider = str(data.get("provider", "auto")).strip().lower() or "auto"
        if provider not in self.VALID_PROVIDERS:
            provider = "auto"
        data["provider"] = provider

        searxng_base_url = str(
            os.getenv("FOXFORGE_SEARXNG_URL", str(data.get("searxng_base_url", "http://127.0.0.1:8080")))
        ).strip()
        data["searxng_base_url"] = searxng_base_url.rstrip("/") or "http://127.0.0.1:8080"

        try:
            searxng_timeout_sec = int(data.get("searxng_timeout_sec", 20))
        except (TypeError, ValueError):
            searxng_timeout_sec = 20
        data["searxng_timeout_sec"] = max(3, min(searxng_timeout_sec, 180))

        data["searxng_engines"] = str(data.get("searxng_engines", "")).strip()
        data["searxng_categories"] = str(data.get("searxng_categories", "")).strip()
        data["searxng_language"] = str(data.get("searxng_language", "")).strip()
        try:
            max_results = int(data.get("max_results", 8))
        except (TypeError, ValueError):
            max_results = 8
        data["max_results"] = max(1, min(max_results, 20))
        data["query_expansion_enabled"] = _bool_setting("query_expansion_enabled", True)
        try:
            query_expansion_variants = int(data.get("query_expansion_variants", 4))
        except (TypeError, ValueError):
            query_expansion_variants = 4
        data["query_expansion_variants"] = max(1, min(query_expansion_variants, 8))
        data["query_decomposition_enabled"] = _bool_setting("query_decomposition_enabled", True)
        try:
            query_decomposition_max_sub = int(data.get("query_decomposition_max_sub", 5))
        except (TypeError, ValueError):
            query_decomposition_max_sub = 5
        data["query_decomposition_max_sub"] = max(1, min(query_decomposition_max_sub, 8))
        data["pre_crawl_seed_selection_enabled"] = _bool_setting("pre_crawl_seed_selection_enabled", True)
        try:
            pre_crawl_results_per_query = int(data.get("pre_crawl_results_per_query", 20))
        except (TypeError, ValueError):
            pre_crawl_results_per_query = 20
        # Search providers in this stack currently cap at 20 results/query.
        data["pre_crawl_results_per_query"] = max(20, min(pre_crawl_results_per_query, 20))
        try:
            pre_crawl_primary_quota = int(data.get("pre_crawl_primary_quota", 5))
        except (TypeError, ValueError):
            pre_crawl_primary_quota = 5
        data["pre_crawl_primary_quota"] = max(1, min(pre_crawl_primary_quota, 12))
        try:
            pre_crawl_extra_quota_min = int(data.get("pre_crawl_extra_quota_min", 2))
        except (TypeError, ValueError):
            pre_crawl_extra_quota_min = 2
        data["pre_crawl_extra_quota_min"] = max(1, min(pre_crawl_extra_quota_min, 6))
        try:
            pre_crawl_extra_quota_max = int(data.get("pre_crawl_extra_quota_max", 3))
        except (TypeError, ValueError):
            pre_crawl_extra_quota_max = 3
        pre_crawl_extra_quota_max = max(1, min(pre_crawl_extra_quota_max, 8))
        data["pre_crawl_extra_quota_max"] = max(data["pre_crawl_extra_quota_min"], pre_crawl_extra_quota_max)
        data["smart_query_variants_enabled"] = _bool_setting("smart_query_variants_enabled", True)
        try:
            smart_query_variants_limit = int(data.get("smart_query_variants_limit", 3))
        except (TypeError, ValueError):
            smart_query_variants_limit = 3
        data["smart_query_variants_limit"] = max(1, min(smart_query_variants_limit, 6))
        try:
            smart_query_summary_chars = int(data.get("smart_query_summary_chars", 2200))
        except (TypeError, ValueError):
            smart_query_summary_chars = 2200
        data["smart_query_summary_chars"] = max(600, min(smart_query_summary_chars, 12000))
        try:
            smart_query_cache_rows = int(data.get("smart_query_cache_rows", 6))
        except (TypeError, ValueError):
            smart_query_cache_rows = 6
        data["smart_query_cache_rows"] = max(0, min(smart_query_cache_rows, 20))
        data["iterative_search_enabled"] = _bool_setting("iterative_search_enabled", True)
        try:
            iterative_search_time_budget_sec = float(data.get("iterative_search_time_budget_sec", 25))
        except (TypeError, ValueError):
            iterative_search_time_budget_sec = 25.0
        data["iterative_search_time_budget_sec"] = max(5.0, min(iterative_search_time_budget_sec, 180.0))
        data["embedding_content_filter_enabled"] = _bool_setting("embedding_content_filter_enabled", True)
        data["source_scoring_enabled"] = _bool_setting("source_scoring_enabled", True)
        try:
            min_quality_sources = int(data.get("min_quality_sources", 2))
        except (TypeError, ValueError):
            min_quality_sources = 2
        data["min_quality_sources"] = max(1, min(min_quality_sources, 8))
        try:
            context_min_source_score = float(data.get("context_min_source_score", 0.62))
        except (TypeError, ValueError):
            context_min_source_score = 0.62
        data["context_min_source_score"] = max(0.1, min(context_min_source_score, 1.0))
        data["fresh_runs_enabled"] = _bool_setting("fresh_runs_enabled", True)
        try:
            fresh_runs_history_limit = int(data.get("fresh_runs_history_limit", 6))
        except (TypeError, ValueError):
            fresh_runs_history_limit = 6
        data["fresh_runs_history_limit"] = max(1, min(fresh_runs_history_limit, 20))
        try:
            fresh_runs_min_new_domains = int(data.get("fresh_runs_min_new_domains", 4))
        except (TypeError, ValueError):
            fresh_runs_min_new_domains = 4
        data["fresh_runs_min_new_domains"] = max(1, min(fresh_runs_min_new_domains, 12))
        data["conflict_detection_enabled"] = _bool_setting("conflict_detection_enabled", True)
        data["crawl_relevance_gating_enabled"] = _bool_setting("crawl_relevance_gating_enabled", False)
        try:
            crawl_relevance_min_score = float(data.get("crawl_relevance_min_score", 0.1))
        except (TypeError, ValueError):
            crawl_relevance_min_score = 0.1
        data["crawl_relevance_min_score"] = max(0.0, min(crawl_relevance_min_score, 1.0))
        data["crawl_enabled"] = _bool_setting("crawl_enabled", True)

        try:
            crawl_depth = int(data.get("crawl_depth", 2))
        except (TypeError, ValueError):
            crawl_depth = 2
        data["crawl_depth"] = max(0, min(crawl_depth, 4))

        try:
            crawl_max_pages = int(data.get("crawl_max_pages", 18))
        except (TypeError, ValueError):
            crawl_max_pages = 18
        data["crawl_max_pages"] = max(1, min(crawl_max_pages, 80))

        try:
            crawl_links_per_page = int(data.get("crawl_links_per_page", 8))
        except (TypeError, ValueError):
            crawl_links_per_page = 8
        data["crawl_links_per_page"] = max(1, min(crawl_links_per_page, 30))

        try:
            crawl_timeout_sec = int(data.get("crawl_timeout_sec", 0))
        except (TypeError, ValueError):
            crawl_timeout_sec = 0
        data["crawl_timeout_sec"] = max(0, min(crawl_timeout_sec, 180))

        data["crawl4ai_enabled"] = _bool_setting("crawl4ai_enabled", True)
        data["crawl4ai_base_url"] = (
            str(os.getenv("FOXFORGE_CRAWL4AI_URL", str(data.get("crawl4ai_base_url", "http://127.0.0.1:11235"))))
            .strip()
            .rstrip("/")
        )

        try:
            crawl4ai_timeout_sec = int(data.get("crawl4ai_timeout_sec", 40))
        except (TypeError, ValueError):
            crawl4ai_timeout_sec = 40
        data["crawl4ai_timeout_sec"] = max(3, min(crawl4ai_timeout_sec, 300))

        try:
            crawl4ai_retry_attempts = int(data.get("crawl4ai_retry_attempts", 2))
        except (TypeError, ValueError):
            crawl4ai_retry_attempts = 2
        data["crawl4ai_retry_attempts"] = max(1, min(crawl4ai_retry_attempts, 8))
        data["crawl4ai_css_selector"] = str(data.get("crawl4ai_css_selector", "article,main,p")).strip()

        data["newspaper_enabled"] = _bool_setting("newspaper_enabled", True)
        data["newspaper_language"] = str(data.get("newspaper_language", "")).strip()

        try:
            search_retry_attempts = int(data.get("search_retry_attempts", 3))
        except (TypeError, ValueError):
            search_retry_attempts = 3
        data["search_retry_attempts"] = max(1, min(search_retry_attempts, 8))

        try:
            crawl_retry_attempts = int(data.get("crawl_retry_attempts", 3))
        except (TypeError, ValueError):
            crawl_retry_attempts = 3
        data["crawl_retry_attempts"] = max(1, min(crawl_retry_attempts, 8))

        data["crawl_same_domain_only"] = _bool_setting("crawl_same_domain_only", True)

        try:
            crawl_text_chars = int(data.get("crawl_text_chars", 2500))
        except (TypeError, ValueError):
            crawl_text_chars = 2500
        data["crawl_text_chars"] = max(250, min(crawl_text_chars, 6000))

        # TOR proxy settings (disabled by default — enable when TOR daemon is running)
        data["tor_proxy_enabled"] = _bool_setting("tor_proxy_enabled", False)
        data["tor_proxy_url"] = str(data.get("tor_proxy_url", "socks5h://127.0.0.1:9050")).strip()
        try:
            tor_timeout_multiplier = float(data.get("tor_timeout_multiplier", 2.5))
        except (TypeError, ValueError):
            tor_timeout_multiplier = 2.5
        data["tor_timeout_multiplier"] = max(1.0, min(tor_timeout_multiplier, 10.0))

        # Cache / retention settings
        try:
            cache_ttl_days = int(data.get("cache_ttl_days", 14))
        except (TypeError, ValueError):
            cache_ttl_days = 14
        data["cache_ttl_days"] = max(1, min(cache_ttl_days, 365))

        try:
            log_retain_days = int(data.get("log_retain_days", 30))
        except (TypeError, ValueError):
            log_retain_days = 30
        data["log_retain_days"] = max(1, min(log_retain_days, 365))

        return data

    def _save_settings(self, settings: dict[str, Any]) -> None:
        self.settings_path.write_text(json.dumps(settings, indent=2, ensure_ascii=True), encoding="utf-8")

    def _urlopen(
        self,
        req: urllib.request.Request,
        timeout: int,
        *,
        use_tor: bool | None = None,
        settings: dict[str, Any] | None = None,
    ) -> Any:
        """Wrapper around urllib.request.urlopen with optional TOR proxy routing.

        When TOR is active (via use_tor=True or self._tor_active) and
        tor_proxy_enabled is set in settings, routes the request through a
        SOCKS5 proxy. The socks5h:// scheme resolves DNS through the proxy to
        prevent leaks — critical for .onion domains.

        Requires PySocks (pip install PySocks) for SOCKS5 support.
        """
        try:
            routing = load_model_routing(self.repo_root)
            use_fetch_mcp = bool(routing.get("mcp.use_fetch", False)) if isinstance(routing, dict) else False
        except Exception:
            use_fetch_mcp = False
        if use_fetch_mcp:
            try:
                method = str(req.get_method() or "GET").upper()
            except Exception:
                method = "GET"
            if method == "GET":
                try:
                    from shared_tools.mcp_client import mcp_fetch_url

                    body = mcp_fetch_url(self.repo_root, str(req.full_url or ""), timeout_sec=max(int(timeout or 0), 1))
                except Exception:
                    body = None
                if isinstance(body, (bytes, bytearray)):
                    return _InMemoryResponse(bytes(body))

        _tor = use_tor if use_tor is not None else self._tor_active
        if _tor:
            _s = settings or self._load_settings()
            if _s.get("tor_proxy_enabled", False):
                proxy_url = str(_s.get("tor_proxy_url", "socks5h://127.0.0.1:9050")).strip()
                multiplier = float(_s.get("tor_timeout_multiplier", 2.5))
                timeout = int(timeout * multiplier)
                proxy_handler = urllib.request.ProxyHandler({
                    "http": proxy_url,
                    "https": proxy_url,
                })
                opener = urllib.request.build_opener(proxy_handler)
                return opener.open(req, timeout=max(timeout, 1))
        if timeout <= 0:
            return urllib.request.urlopen(req)
        return urllib.request.urlopen(req, timeout=timeout)

    def get_mode(self) -> str:
        with self.lock:
            settings = self._load_settings()
            return str(settings.get("mode", "auto"))

    def set_mode(self, mode: str) -> str:
        key = mode.strip().lower()
        if key not in self.VALID_MODES:
            raise ValueError("Invalid web mode. Use: off, ask, auto.")
        with self.lock:
            settings = self._load_settings()
            settings["mode"] = key
            self._save_settings(settings)
        return key

    def mode_text(self) -> str:
        settings = self._load_settings()
        return (
            "Web research mode:\n"
            f"- mode: {settings.get('mode', 'auto')}\n"
            f"- provider: {settings.get('provider', 'auto')}\n"
            f"- max_results: {settings.get('max_results', 8)}\n"
            f"- query_expansion_enabled: {settings.get('query_expansion_enabled', True)}\n"
            f"- query_expansion_variants: {settings.get('query_expansion_variants', 4)}\n"
            f"- query_decomposition_enabled: {settings.get('query_decomposition_enabled', True)}\n"
            f"- query_decomposition_max_sub: {settings.get('query_decomposition_max_sub', 5)}\n"
            f"- pre_crawl_seed_selection_enabled: {settings.get('pre_crawl_seed_selection_enabled', True)}\n"
            f"- pre_crawl_results_per_query: {settings.get('pre_crawl_results_per_query', 20)}\n"
            f"- pre_crawl_primary_quota: {settings.get('pre_crawl_primary_quota', 5)}\n"
            f"- pre_crawl_extra_quota_min: {settings.get('pre_crawl_extra_quota_min', 2)}\n"
            f"- pre_crawl_extra_quota_max: {settings.get('pre_crawl_extra_quota_max', 3)}\n"
            f"- smart_query_variants_enabled: {settings.get('smart_query_variants_enabled', True)}\n"
            f"- smart_query_variants_limit: {settings.get('smart_query_variants_limit', 3)}\n"
            f"- smart_query_summary_chars: {settings.get('smart_query_summary_chars', 2200)}\n"
            f"- smart_query_cache_rows: {settings.get('smart_query_cache_rows', 6)}\n"
            f"- iterative_search_enabled: {settings.get('iterative_search_enabled', True)}\n"
            f"- iterative_search_time_budget_sec: {settings.get('iterative_search_time_budget_sec', 25)}\n"
            f"- embedding_content_filter_enabled: {settings.get('embedding_content_filter_enabled', True)}\n"
            f"- source_scoring_enabled: {settings.get('source_scoring_enabled', False)}\n"
            f"- min_quality_sources: {settings.get('min_quality_sources', 2)}\n"
            f"- context_min_source_score: {settings.get('context_min_source_score', 0.62)}\n"
            f"- fresh_runs_enabled: {settings.get('fresh_runs_enabled', True)}\n"
            f"- fresh_runs_history_limit: {settings.get('fresh_runs_history_limit', 6)}\n"
            f"- fresh_runs_min_new_domains: {settings.get('fresh_runs_min_new_domains', 4)}\n"
            f"- conflict_detection_enabled: {settings.get('conflict_detection_enabled', False)}\n"
            f"- crawl_relevance_gating_enabled: {settings.get('crawl_relevance_gating_enabled', False)}\n"
            f"- crawl_relevance_min_score: {settings.get('crawl_relevance_min_score', 0.1)}\n"
            f"- searxng_base_url: {settings.get('searxng_base_url', 'http://127.0.0.1:8080')}\n"
            f"- searxng_timeout_sec: {settings.get('searxng_timeout_sec', 20)}\n"
            f"- searxng_engines: {settings.get('searxng_engines', '') or '(auto)'}\n"
            f"- searxng_categories: {settings.get('searxng_categories', '') or '(auto)'}\n"
            f"- searxng_language: {settings.get('searxng_language', '') or '(auto)'}\n"
            f"- crawl_enabled: {settings.get('crawl_enabled', True)}\n"
            f"- crawl_depth: {settings.get('crawl_depth', 2)}\n"
            f"- crawl_max_pages: {settings.get('crawl_max_pages', 18)}\n"
            f"- crawl_links_per_page: {settings.get('crawl_links_per_page', 8)}\n"
            f"- crawl_timeout_sec: {settings.get('crawl_timeout_sec', 12)}\n"
            f"- crawl4ai_enabled: {settings.get('crawl4ai_enabled', True)}\n"
            f"- crawl4ai_base_url: {settings.get('crawl4ai_base_url', 'http://127.0.0.1:11235')}\n"
            f"- crawl4ai_timeout_sec: {settings.get('crawl4ai_timeout_sec', 40)}\n"
            f"- crawl4ai_retry_attempts: {settings.get('crawl4ai_retry_attempts', 2)}\n"
            f"- crawl4ai_css_selector: {settings.get('crawl4ai_css_selector', '') or '(default)'}\n"
            f"- newspaper_enabled: {settings.get('newspaper_enabled', True)}\n"
            f"- newspaper_language: {settings.get('newspaper_language', '') or '(auto)'}\n"
            f"- search_retry_attempts: {settings.get('search_retry_attempts', 3)}\n"
            f"- crawl_retry_attempts: {settings.get('crawl_retry_attempts', 3)}\n"
            f"- crawl_same_domain_only: {settings.get('crawl_same_domain_only', True)}\n"
            f"- crawl_text_chars: {settings.get('crawl_text_chars', 800)}"
        )

    def get_provider(self) -> str:
        with self.lock:
            settings = self._load_settings()
            return str(settings.get("provider", "auto"))

    def provider_text(self) -> str:
        settings = self._load_settings()
        return (
            "Web research provider:\n"
            f"- provider: {settings.get('provider', 'auto')}\n"
            f"- mode: {settings.get('mode', 'auto')}\n"
            f"- searxng_base_url: {settings.get('searxng_base_url', 'http://127.0.0.1:8080')}"
        )

    def set_provider(self, provider: str) -> str:
        key = provider.strip().lower()
        if key not in self.VALID_PROVIDERS:
            raise ValueError("Invalid web provider. Use: auto, searxng, duckduckgo_html, duckduckgo_api.")
        with self.lock:
            settings = self._load_settings()
            settings["provider"] = key
            self._save_settings(settings)
        return key

    def create_pending(self, *, project: str, lane: str, query: str, reason: str, topic_type: str = "general") -> dict[str, Any]:
        query_text = query.strip()
        if not query_text:
            raise ValueError("Web pending query cannot be empty.")
        normalized_topic_type = str(topic_type or "").strip().lower() or "general"
        row = {
            "id": f"web_{uuid.uuid4().hex[:8]}",
            "type": "web_research",
            "status": "open",
            "project": project.strip() or "general",
            "lane": lane.strip() or "project",
            "topic_type": normalized_topic_type,
            "query": query_text,
            "reason": reason.strip() or "Web freshness/citation check requested.",
            "question": "Allow live web research for this request?",
            "summary": (
                f"Query: {query_text[:220]}"
                + ("" if len(query_text) <= 220 else "...")
                + f" | Reason: {(reason.strip() or 'Web freshness/citation check requested.')[:160]}"
            ),
            "created_at": _now_iso(),
            "updated_at": _now_iso(),
        }
        with self.lock:
            rows = self._load_pending()
            rows.append(row)
            self._save_pending(rows)
        return row

    def list_pending(self, limit: int = 50) -> list[dict[str, Any]]:
        limit = max(1, min(limit, 500))
        with self.lock:
            rows = [x for x in self._load_pending() if str(x.get("status", "")).lower() == "open"]
        rows.sort(key=lambda x: str(x.get("created_at", "")), reverse=True)
        return rows[:limit]

    def get_request(self, request_id: str) -> dict[str, Any] | None:
        key = request_id.strip()
        with self.lock:
            rows = self._load_pending()
            for row in rows:
                if str(row.get("id", "")) == key:
                    return row
        return None

    def ignore(self, request_id: str, reason: str = "") -> dict[str, Any] | None:
        key = request_id.strip()
        with self.lock:
            rows = self._load_pending()
            hit: dict[str, Any] | None = None
            for row in rows:
                if str(row.get("id", "")) != key:
                    continue
                if str(row.get("status", "")).lower() != "open":
                    return None
                row["status"] = "ignored"
                row["ignore_reason"] = reason.strip() or "ignored by user"
                row["updated_at"] = _now_iso()
                row["resolved_at"] = _now_iso()
                hit = row
                break
            if hit is None:
                return None
            self._save_pending(rows)
            return hit

    def mark_routed(self, request_id: str, *, target: str, note: str = "", handoff_id: str = "") -> dict[str, Any] | None:
        key = request_id.strip()
        with self.lock:
            rows = self._load_pending()
            hit: dict[str, Any] | None = None
            for row in rows:
                if str(row.get("id", "")) != key:
                    continue
                if str(row.get("status", "")).lower() != "open":
                    return None
                row["status"] = "routed_external"
                row["routed_target"] = target.strip().lower()
                row["routed_note"] = note.strip()
                row["handoff_id"] = handoff_id.strip()
                row["updated_at"] = _now_iso()
                row["resolved_at"] = _now_iso()
                hit = row
                break
            if hit is None:
                return None
            self._save_pending(rows)
            return hit

    def _strip_tags(self, text: str) -> str:
        return re.sub(r"<[^>]+>", "", text)

    def _unwrap_duckduckgo_url(self, href: str) -> str:
        ref = html.unescape(href.strip())
        if ref.startswith("//"):
            ref = "https:" + ref
        if ref.startswith("/"):
            ref = "https://duckduckgo.com" + ref
        parsed = urllib.parse.urlsplit(ref)
        if "duckduckgo.com" in parsed.netloc and parsed.path.startswith("/l/"):
            params = urllib.parse.parse_qs(parsed.query)
            uddg = params.get("uddg", [])
            if uddg:
                return urllib.parse.unquote(uddg[0])
        return ref

    def _search_duckduckgo_html(self, query: str, max_results: int) -> list[dict[str, str]]:
        encoded = urllib.parse.urlencode({"q": query})
        url = f"https://duckduckgo.com/html/?{encoded}"
        req = urllib.request.Request(
            url=url,
            headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Foxforge/1.0",
            },
            method="GET",
        )
        with self._urlopen(req, timeout=25) as resp:
            body = resp.read().decode("utf-8", errors="ignore")

        pattern = re.compile(
            r'<a[^>]*class="[^"]*result__a[^"]*"[^>]*href="([^"]+)"[^>]*>(.*?)</a>',
            re.IGNORECASE | re.DOTALL,
        )
        out: list[dict[str, str]] = []
        seen: set[str] = set()
        for href, title_html in pattern.findall(body):
            url_value = self._unwrap_duckduckgo_url(href)
            if not url_value.startswith("http://") and not url_value.startswith("https://"):
                continue
            if url_value in seen:
                continue
            seen.add(url_value)
            title = html.unescape(self._strip_tags(title_html)).strip()
            if not title:
                title = url_value
            out.append({"title": title, "url": url_value, "snippet": ""})
            if len(out) >= max_results:
                break
        return out

    def _search_duckduckgo_api(self, query: str, max_results: int) -> list[dict[str, str]]:
        params = urllib.parse.urlencode(
            {
                "q": query,
                "format": "json",
                "no_redirect": "1",
                "no_html": "1",
            }
        )
        url = f"https://api.duckduckgo.com/?{params}"
        req = urllib.request.Request(
            url=url,
            headers={"User-Agent": "Foxforge/1.0"},
            method="GET",
        )
        with self._urlopen(req, timeout=20) as resp:
            data = json.loads(resp.read().decode("utf-8", errors="ignore"))

        out: list[dict[str, str]] = []
        seen: set[str] = set()

        abstract_url = str(data.get("AbstractURL", "")).strip()
        abstract_text = str(data.get("AbstractText", "")).strip()
        heading = str(data.get("Heading", "")).strip() or abstract_url
        if abstract_url and abstract_url not in seen:
            seen.add(abstract_url)
            out.append({"title": heading, "url": abstract_url, "snippet": abstract_text})

        def _walk_topics(items: Any) -> None:
            if not isinstance(items, list):
                return
            for item in items:
                if len(out) >= max_results:
                    return
                if isinstance(item, dict) and "Topics" in item:
                    _walk_topics(item.get("Topics"))
                    continue
                if not isinstance(item, dict):
                    continue
                url_value = str(item.get("FirstURL", "")).strip()
                text = str(item.get("Text", "")).strip()
                if not url_value or url_value in seen:
                    continue
                seen.add(url_value)
                title = text.split(" - ")[0].strip() if text else url_value
                out.append({"title": title, "url": url_value, "snippet": text})

        _walk_topics(data.get("RelatedTopics"))
        return out[:max_results]

    def _search_searxng(self, query: str, max_results: int, settings: dict[str, Any]) -> list[dict[str, str]]:
        base_url = str(settings.get("searxng_base_url", "http://127.0.0.1:8080")).strip().rstrip("/")
        if not base_url:
            return []
        endpoint = f"{base_url}/search"
        timeout_sec = int(settings.get("searxng_timeout_sec", 20))
        params: dict[str, str] = {
            "q": query,
            "format": "json",
        }
        engines = str(settings.get("searxng_engines", "")).strip()
        categories = str(settings.get("searxng_categories", "")).strip()
        language = str(settings.get("searxng_language", "")).strip()
        if engines:
            params["engines"] = engines
        if categories:
            params["categories"] = categories
        if language:
            params["language"] = language
        query_string = urllib.parse.urlencode(params)
        req = urllib.request.Request(
            url=f"{endpoint}?{query_string}",
            headers={"User-Agent": "Foxforge/1.0"},
            method="GET",
        )
        with self._urlopen(req, timeout=timeout_sec) as resp:
            payload = json.loads(resp.read().decode("utf-8", errors="ignore"))
        rows = payload.get("results", [])
        if not isinstance(rows, list):
            return []
        out: list[dict[str, str]] = []
        seen: set[str] = set()
        for row in rows:
            if not isinstance(row, dict):
                continue
            url_value = self._normalize_url(str(row.get("url", "")).strip())
            if not url_value or url_value in seen:
                continue
            seen.add(url_value)
            title = str(row.get("title", "")).strip() or url_value
            snippet = str(row.get("content", "")).strip() or str(row.get("snippet", "")).strip()
            out.append({"title": title, "url": url_value, "snippet": snippet})
            if len(out) >= max_results:
                break
        return out

    def _merge_results(
        self,
        primary: list[dict[str, str]],
        secondary: list[dict[str, str]],
        limit: int,
    ) -> list[dict[str, str]]:
        out: list[dict[str, str]] = []
        seen: set[str] = set()
        for row in list(primary) + list(secondary):
            if len(out) >= limit:
                break
            if not isinstance(row, dict):
                continue
            url_value = self._normalize_url(str(row.get("url", "")).strip())
            if not url_value or url_value in seen:
                continue
            seen.add(url_value)
            payload: dict[str, str] = {
                "title": str(row.get("title", "")).strip() or url_value,
                "url": url_value,
                "snippet": str(row.get("snippet", "")).strip(),
            }
            query_variant = str(row.get("query_variant", "")).strip()
            if query_variant:
                payload["query_variant"] = query_variant
            out.append(payload)
        return out[:limit]

    def _merge_query_lists(self, primary: list[str], secondary: list[str]) -> list[str]:
        out: list[str] = []
        seen: set[str] = set()
        for value in list(primary) + list(secondary):
            text = " ".join(str(value or "").split()).strip()
            if not text:
                continue
            key = text.lower()
            if key in seen:
                continue
            seen.add(key)
            out.append(text)
        return out

    def _is_quick_lookup_note(self, note: str) -> bool:
        return "quick_chat_lookup" in str(note or "").strip().lower()

    def _recent_source_domains_for_project(self, project: str, limit: int = 6) -> set[str]:
        rows = self.recent_sources_for_project(project, limit=max(1, min(limit, 30)))
        domains: set[str] = set()
        for row in rows:
            sources = row.get("sources", [])
            if not isinstance(sources, list):
                continue
            for source in sources:
                if not isinstance(source, dict):
                    continue
                domain = str(source.get("source_domain", "") or source.get("domain", "")).strip().lower()
                if not domain:
                    domain = self._hostname(str(source.get("url", "")).strip())
                if domain:
                    domains.add(domain)
        return domains

    def _recent_project_queries(self, project: str, limit: int = 6) -> list[str]:
        rows = self.recent_sources_for_project(project, limit=max(1, min(limit, 30)))
        queries: list[str] = []
        seen: set[str] = set()
        for row in rows:
            query = " ".join(str(row.get("query", "")).split()).strip()
            if not query:
                continue
            key = query.lower()
            if key in seen:
                continue
            seen.add(key)
            queries.append(query)
            if len(queries) >= limit:
                break
        return queries

    def _latest_research_summary_excerpt(self, project: str, max_chars: int = 2200) -> str:
        slug = str(project or "").strip() or "general"
        root = self.repo_root / "Projects" / slug / "research_summaries"
        if not root.exists():
            return ""
        try:
            files = sorted(
                [p for p in root.glob("*.md") if p.is_file()],
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
        except Exception:
            return ""
        if not files:
            return ""
        try:
            body = files[0].read_text(encoding="utf-8", errors="ignore")
        except Exception:
            return ""
        text = " ".join(str(body or "").split()).strip()
        if not text:
            return ""
        return text[: max(600, min(int(max_chars or 2200), 12000))]

    def _cached_query_hints(self, project: str, limit: int = 6) -> list[str]:
        hints: list[str] = []
        seen: set[str] = set()
        try:
            rows = self._query_cached_chunks(project, limit=max(1, min(limit, 20)))
        except Exception:
            return hints
        for row in rows:
            if not isinstance(row, dict):
                continue
            title = " ".join(str(row.get("title", "")).split()).strip()
            domain = str(row.get("domain", "")).strip().lower()
            url_value = str(row.get("url", "")).strip()
            if not domain and url_value:
                domain = self._hostname(url_value)
            hint = title or domain or url_value
            hint = hint[:120].strip()
            if not hint:
                continue
            key = hint.lower()
            if key in seen:
                continue
            seen.add(key)
            hints.append(hint)
            if len(hints) >= limit:
                break
        return hints

    _SMART_QUERY_SYSTEM = (
        "You are a web query planner for iterative research runs. "
        "Generate concise, search-ready queries that target NEW evidence and unresolved gaps.\n\n"
        "RULES:\n"
        "- Every query must name the specific entity, product, version, or claim being researched. "
        "Never write a query that could apply to any topic (e.g. 'official documentation', "
        "'release notes', 'version compatibility' alone are forbidden — they must be combined with "
        "the specific subject).\n"
        "- Include at least one query that targets community, practitioner, or real-world perspective "
        "(forums, issue trackers, benchmarks, case studies) rather than only official sources.\n"
        "- Do not repeat near-identical prior queries. Each variant must approach the topic from a "
        "distinct angle: e.g. primary facts, limitations/edge-cases, comparisons, recent changes.\n"
        "Output ONLY the queries, one per line, no numbering or commentary."
    )

    _SMART_QUERY_TOPIC_HINTS: dict[str, str] = {
        "technical": (
            "Prefer queries targeting changelogs, GitHub issues, migration guides, "
            "community benchmarks, and known limitations."
        ),
        "financial": (
            "Prefer queries targeting earnings reports, analyst commentary, SEC filings, "
            "and real-money community discussion (not promotional content)."
        ),
        "medical": (
            "Prefer queries targeting clinical studies, trial registries, peer-reviewed journals, "
            "and practitioner forums rather than health news aggregators."
        ),
        "legal": (
            "Prefer queries targeting primary legal sources (case law, statutes, regulatory filings) "
            "and law review commentary."
        ),
        "current_events": (
            "Prefer queries targeting wire reports, primary government/official statements, "
            "and regional outlets that may have on-the-ground coverage."
        ),
    }

    def _smart_query_variants(
        self,
        *,
        project: str,
        query: str,
        settings: dict[str, Any],
        existing_queries: list[str],
        recent_domains: set[str] | None = None,
        topic_type: str = "",
    ) -> list[str]:
        if not bool(settings.get("smart_query_variants_enabled", True)):
            return []
        base = " ".join(str(query or "").split()).strip()
        if not base:
            return []
        try:
            variant_limit = int(settings.get("smart_query_variants_limit", 3))
        except (TypeError, ValueError):
            variant_limit = 3
        variant_limit = max(1, min(variant_limit, 6))
        try:
            cache_rows = int(settings.get("smart_query_cache_rows", 6))
        except (TypeError, ValueError):
            cache_rows = 6
        recent_queries = self._recent_project_queries(project, limit=6)
        cache_hints = self._cached_query_hints(project, limit=cache_rows)
        recent_domain_list = sorted([d for d in (recent_domains or set()) if d])[:16]
        context_chunks: list[str] = []
        if recent_queries:
            context_chunks.append("Recent run queries:\n- " + "\n- ".join(recent_queries))
        if cache_hints:
            context_chunks.append("Cached source hints:\n- " + "\n- ".join(cache_hints))
        if recent_domain_list:
            context_chunks.append("Recently used domains to diversify away from when possible:\n- " + "\n- ".join(recent_domain_list))
        context_block = "\n\n".join(context_chunks)
        _topic_key = str(topic_type or "").strip().lower()
        _topic_hint = self._SMART_QUERY_TOPIC_HINTS.get(_topic_key, "")
        user_prompt = (
            f"Current user query:\n{base}\n\n"
            f"Already-generated query variants:\n- " + "\n- ".join(existing_queries[:10]) + "\n\n"
            "Create up to "
            f"{variant_limit} additional web queries for this run that are specific, evidence-seeking, and non-redundant."
        )
        if _topic_hint:
            user_prompt += f"\n\nTopic guidance: {_topic_hint}"
        if context_block:
            user_prompt += f"\n\nProject context:\n{context_block}"
        try:
            router = InferenceRouter(self.repo_root)
            cfg = lane_model_config(self.repo_root, "web_query_refiner") or {}
            raw = router.chat(
                model=str(cfg.get("model", "qwen3:8b")),
                system_prompt=self._SMART_QUERY_SYSTEM,
                user_prompt=user_prompt,
                temperature=0.0,
                num_ctx=4096,
                think=False,
                timeout=25,
                retry_attempts=1,
                retry_backoff_sec=0.5,
                fallback_models=cfg.get("fallback_models", ["deepseek-r1:8b"]),
            )
        except Exception:
            return []
        existing_keys = {str(q or "").strip().lower() for q in existing_queries}
        out: list[str] = []
        for line in str(raw or "").splitlines():
            text = line.strip().lstrip("•-–*0123456789.) ")
            text = " ".join(text.split()).strip()
            if not text or len(text) < 6:
                continue
            key = text.lower()
            if key in existing_keys:
                continue
            if key in {x.lower() for x in out}:
                continue
            out.append(text)
            if len(out) >= variant_limit:
                break
        return out

    def _prioritize_seed_domains_for_freshness(
        self,
        seeds: list[dict[str, str]],
        recent_domains: set[str],
        *,
        min_new_domains: int = 4,
    ) -> tuple[list[dict[str, str]], dict[str, Any]]:
        if not seeds:
            return [], {
                "recent_domains_considered": len(recent_domains),
                "novel_domains": 0,
                "novel_seed_rows": 0,
                "strict_novel_only": False,
            }
        novel_rows: list[dict[str, str]] = []
        repeated_rows: list[dict[str, str]] = []
        novel_domains: set[str] = set()
        for row in seeds:
            host = self._hostname(str(row.get("url", "")).strip())
            if host and host in recent_domains:
                repeated_rows.append(row)
                continue
            novel_rows.append(row)
            if host:
                novel_domains.add(host)
        # Favor domain diversity first (one row per fresh domain), then remainder.
        prioritized_novel: list[dict[str, str]] = []
        seen_domain_once: set[str] = set()
        novel_tail: list[dict[str, str]] = []
        for row in novel_rows:
            host = self._hostname(str(row.get("url", "")).strip())
            if not host or host in seen_domain_once:
                novel_tail.append(row)
                continue
            seen_domain_once.add(host)
            prioritized_novel.append(row)
        prioritized_novel.extend(novel_tail)
        strict_novel_only = len(novel_domains) >= max(1, int(min_new_domains or 1))
        ordered = prioritized_novel if strict_novel_only else (prioritized_novel + repeated_rows)
        stats = {
            "recent_domains_considered": len(recent_domains),
            "novel_domains": len(novel_domains),
            "novel_seed_rows": len(novel_rows),
            "strict_novel_only": bool(strict_novel_only),
        }
        return ordered, stats

    def _select_pre_crawl_seeds(
        self,
        *,
        seeds: list[dict[str, str]],
        query: str,
        variant_queries: list[str],
        settings: dict[str, Any],
        resolved_topic: str,
    ) -> tuple[list[dict[str, str]], dict[str, Any]]:
        enabled = bool(settings.get("pre_crawl_seed_selection_enabled", True))
        stats: dict[str, Any] = {
            "enabled": enabled,
            "results_per_query": int(settings.get("pre_crawl_results_per_query", 20) or 20),
            "primary_quota": int(settings.get("pre_crawl_primary_quota", 5) or 5),
            "extra_quota_min": int(settings.get("pre_crawl_extra_quota_min", 2) or 2),
            "extra_quota_max": int(settings.get("pre_crawl_extra_quota_max", 3) or 3),
            "seed_count_before": len(seeds),
            "seed_count_after": len(seeds),
            "selected_by_variant": [],
            "primary_variant": "",
        }
        if not enabled or not seeds:
            return seeds, stats

        ordered_variants = [q for q in variant_queries if str(q or "").strip()]
        variant_seen: set[str] = {str(v).strip().lower() for v in ordered_variants}
        if not ordered_variants:
            ordered_variants = ["main_query"]
        primary_variant = ordered_variants[0]
        stats["primary_variant"] = primary_variant

        query_terms = self._query_terms(query)
        tier_rank = {"tier1": 0, "tier2": 1, "tier3": 2, "banned_sources": 3}
        scored_rows: list[dict[str, Any]] = []
        by_variant: dict[str, list[dict[str, Any]]] = {}
        for row in seeds:
            scored = self._score_one_source(
                row,
                query_terms,
                query=query,
                topic_type=resolved_topic,
            )
            variant = str(scored.get("query_variant", "")).strip() or primary_variant
            if variant.lower() not in variant_seen:
                variant = "__unassigned__"
            scored["query_variant"] = variant
            scored_rows.append(scored)
            by_variant.setdefault(variant, []).append(scored)

        def _rank_key(row: dict[str, Any]) -> tuple[int, int, float, float, int, int]:
            tier = str(row.get("source_tier", "tier3")).strip().lower()
            return (
                1 if bool(row.get("quality_blocked", False)) else 0,
                tier_rank.get(tier, 2),
                -float(row.get("source_score", 0.0)),
                -float(row.get("freshness_score", 0.0)),
                -int(row.get("query_term_hits", 0)),
                -len(str(row.get("snippet", ""))),
            )

        for rows in by_variant.values():
            rows.sort(key=_rank_key)

        selected: list[dict[str, str]] = []
        selected_urls: set[str] = set()

        def _pick_for_variant(variant: str, limit: int) -> int:
            if limit <= 0:
                return 0
            count = 0
            for row in by_variant.get(variant, []):
                url_value = self._normalize_url(str(row.get("url", "")).strip())
                if not url_value or url_value in selected_urls:
                    continue
                if bool(row.get("quality_blocked", False)):
                    continue
                # Skip tier3 seeds with suspiciously short snippets — likely stubs,
                # redirects, or login walls. Tier1/tier2 get authority override.
                _tier = str(row.get("source_tier", "tier3")).strip().lower()
                if _tier == "tier3" and len(str(row.get("snippet", "")).strip()) < 35:
                    continue
                selected_urls.add(url_value)
                payload: dict[str, str] = {
                    "title": str(row.get("title", "")).strip() or url_value,
                    "url": url_value,
                    "snippet": str(row.get("snippet", "")).strip(),
                    "query_variant": variant,
                }
                selected.append(payload)
                count += 1
                if count >= limit:
                    break
            return count

        primary_quota = max(1, int(stats["primary_quota"]))
        extra_quota_min = max(1, int(stats["extra_quota_min"]))
        extra_quota_max = max(extra_quota_min, int(stats["extra_quota_max"]))
        selected_by_variant: list[dict[str, Any]] = []

        picked_primary = _pick_for_variant(primary_variant, primary_quota)
        selected_by_variant.append(
            {
                "variant": primary_variant,
                "quota": primary_quota,
                "selected": picked_primary,
                "available": len(by_variant.get(primary_variant, [])),
            }
        )

        extra_variants = ordered_variants[1:]
        for variant in extra_variants:
            picked = _pick_for_variant(variant, extra_quota_max)
            selected_by_variant.append(
                {
                    "variant": variant,
                    "quota_min": extra_quota_min,
                    "quota_max": extra_quota_max,
                    "selected": picked,
                    "available": len(by_variant.get(variant, [])),
                }
            )

        # Floor fill: try to satisfy at least (5 + 2 per extra variant) when possible.
        minimum_target = primary_quota + (extra_quota_min * len(extra_variants))
        if len(selected) < minimum_target:
            spillover = sorted(scored_rows, key=_rank_key)
            for row in spillover:
                if len(selected) >= minimum_target:
                    break
                if bool(row.get("quality_blocked", False)):
                    continue
                _spill_tier = str(row.get("source_tier", "tier3")).strip().lower()
                if _spill_tier == "tier3" and len(str(row.get("snippet", "")).strip()) < 35:
                    continue
                url_value = self._normalize_url(str(row.get("url", "")).strip())
                if not url_value or url_value in selected_urls:
                    continue
                variant = str(row.get("query_variant", "")).strip() or primary_variant
                selected_urls.add(url_value)
                selected.append(
                    {
                        "title": str(row.get("title", "")).strip() or url_value,
                        "url": url_value,
                        "snippet": str(row.get("snippet", "")).strip(),
                        "query_variant": variant,
                    }
                )

        if not selected:
            # Never return empty when seeds existed.
            selected = self._merge_results([], seeds, min(len(seeds), max(1, primary_quota)))

        stats["selected_by_variant"] = selected_by_variant
        stats["seed_count_after"] = len(selected)
        return selected, stats

    def _resolve_topic_type(self, query: str, topic_type: str) -> str:
        base = str(topic_type or "").strip().lower() or "general"
        mapped_base = self.TOPIC_FAMILY_ALIASES.get(base, base)
        detected = str(detect_topic_type(query, mapped_base) or mapped_base).strip().lower() or "general"
        if (
            detected in {"combat_sports", "sports_event"}
            and mapped_base not in {"general", "sports", "current_events"}
        ):
            return mapped_base
        return self.TOPIC_FAMILY_ALIASES.get(detected, detected)

    _DECOMPOSE_SYSTEM = (
        "You are a search query decomposer. Given a user question, output 1-5 independent "
        "search queries, one per line. If the question asks about multiple specific entities "
        "(brands, products, people, places, prices) create a separate query for each entity. "
        "If it is already a simple single-topic question output just the original query unchanged. "
        "Output ONLY the queries, nothing else. No explanations, no bullets, no numbers."
    )

    def _decompose_query(self, query: str, settings: dict[str, Any], *, max_sub: int = 5) -> list[str]:
        """Use a local LLM to break a compound query into independent sub-queries."""
        base = " ".join(str(query or "").split()).strip()
        if not base or not bool(settings.get("query_decomposition_enabled", True)):
            return [base] if base else []
        try:
            router = InferenceRouter(self.repo_root)
            cfg = lane_model_config(self.repo_root, "web_query_refiner") or {}
            raw = router.chat(
                model=str(cfg.get("model", "qwen3:8b")),
                system_prompt=self._DECOMPOSE_SYSTEM,
                user_prompt=base,
                temperature=0.0,
                num_ctx=512,
                think=False,
                timeout=10,
                retry_attempts=1,
                retry_backoff_sec=0.5,
                fallback_models=cfg.get("fallback_models", ["deepseek-r1:8b"]),
            )
        except Exception:
            return [base]
        lines = [ln.strip().lstrip("•-–*0123456789.) ") for ln in str(raw or "").splitlines()]
        subs = [ln for ln in lines if ln and len(ln) >= 4][:max_sub]
        return subs if subs else [base]

    def _expand_queries(self, query: str, settings: dict[str, Any], topic_type: str = "general") -> list[str]:
        base = " ".join(str(query or "").split()).strip()
        if not base:
            return []
        if not bool(settings.get("query_expansion_enabled", True)):
            return [base]

        try:
            max_variants = int(settings.get("query_expansion_variants", 4))
        except (TypeError, ValueError):
            max_variants = 4
        max_variants = max(1, min(max_variants, 8))

        out: list[str] = []
        seen: set[str] = set()

        def _add(value: str) -> None:
            text = " ".join(str(value or "").split()).strip()
            if not text:
                return
            key = text.lower()
            if key in seen:
                return
            seen.add(key)
            out.append(text)

        resolved_topic = self._resolve_topic_type(base, topic_type)

        # Technical queries: tighter variant cap and suppress low-signal recency/meta
        # suffixes. TOPIC_HINTS for "technical" are already curated (docs, changelogs,
        # version compatibility) so we only want base + 2 hints at most.
        _is_technical = resolved_topic == "technical"
        if _is_technical:
            max_variants = min(max_variants, 3)

        low = base.lower()
        has_recency_hint = any(
            token in low
            for token in (
                "latest",
                "current",
                "recent",
                "today",
                "this week",
                "this month",
                "what's new",
                "whats new",
                "new in",
                "update",
                "updates",
                "news",
            )
        )
        has_year_hint = bool(re.search(r"\b(19|20)\d{2}\b", base))
        current_year = datetime.now(timezone.utc).year

        _add(base)
        for hint in self.TOPIC_HINTS.get(resolved_topic, ()):
            _add(f"{base} {hint}")
        if not _is_technical:
            if not has_recency_hint:
                _add(f"latest {base}")
                _add(f"{base} recent updates")
            if not has_year_hint:
                _add(f"{base} {current_year}")
            _add(f"{base} official announcement")
            _add(f"{base} timeline")
            _add(f"{base} analysis")

        return out[:max_variants]

    def _refine_queries_for_second_pass(
        self,
        query: str,
        settings: dict[str, Any],
        topic_type: str = "general",
    ) -> list[str]:
        base = " ".join(str(query or "").split()).strip()
        if not base:
            return []
        resolved_topic = self._resolve_topic_type(base, topic_type)
        try:
            max_variants = int(settings.get("query_refine_variants", 4))
        except (TypeError, ValueError):
            max_variants = 4
        max_variants = max(1, min(max_variants, 6))

        out: list[str] = []
        seen: set[str] = set()

        def _add(value: str) -> None:
            text = " ".join(str(value or "").split()).strip()
            if not text:
                return
            key = text.lower()
            if key in seen:
                return
            seen.add(key)
            out.append(text)

        event_match = re.search(
            r"\b(?:ufc\s*\d+|fight night\s*\d*|bellator\s*\d+|pfl\s*\d+|one championship|boxing)\b",
            base,
            flags=re.IGNORECASE,
        )
        event_label = event_match.group(0).strip() if event_match else ""

        if resolved_topic == "combat_sports":
            if event_label:
                _add(f"full fight card {event_label}")
                _add(f"{event_label} full fight card")
                _add(f"{event_label} official card")
                _add(f"{event_label} main card prelims")
                _add(f"{event_label} bout order")
            else:
                _add(f"{base} full fight card")
                _add(f"{base} official card")
                _add(f"{base} main card prelims")
                _add(f"{base} bout order")
        elif resolved_topic == "sports_event":
            _add(f"{base} official schedule")
            _add(f"{base} official lineup")
            _add(f"{base} full details")
            _add(f"{base} official preview")
        else:
            _add(f"{base} official")
            _add(f"{base} full details")
            _add(f"{base} exact details")
            _add(f"{base} latest official update")

        return out[:max_variants]

    _FOLLOWUP_SYSTEM = (
        "You are a research gap analyst. Given a search query and the titles/snippets of results "
        "found so far, identify 1-3 specific follow-up searches that would fill gaps. "
        "Focus on missing specifics (prices, dates, specific model names, comparisons) rather "
        "than broad background context. If the results already answer the query well, output NONE. "
        "Output ONLY search queries, one per line. No explanations, no bullets, no numbers."
    )

    def _generate_followup_queries(
        self,
        query: str,
        sources: list[dict[str, Any]],
        settings: dict[str, Any],
    ) -> list[str]:
        """Use a local LLM to generate follow-up searches based on pass-1 source gaps."""
        if not bool(settings.get("iterative_search_enabled", True)):
            return []
        base = " ".join(str(query or "").split()).strip()
        if not base or not sources:
            return []
        # Build compact summary of top-5 sources for the prompt
        top_sources = sources[:5]
        source_lines = []
        for i, s in enumerate(top_sources, 1):
            title = str(s.get("title", "")).strip()[:80]
            snippet = str(s.get("snippet", "")).strip()[:120]
            source_lines.append(f"{i}. {title} — {snippet}")
        source_summary = "\n".join(source_lines)
        user_prompt = f"Query: {base}\n\nSearch results so far:\n{source_summary}"
        try:
            router = InferenceRouter(self.repo_root)
            cfg = lane_model_config(self.repo_root, "web_query_refiner") or {}
            raw = router.chat(
                model=str(cfg.get("model", "qwen3:8b")),
                system_prompt=self._FOLLOWUP_SYSTEM,
                user_prompt=user_prompt,
                temperature=0.0,
                num_ctx=1024,
                think=False,
                timeout=10,
                retry_attempts=1,
                retry_backoff_sec=0.5,
                fallback_models=cfg.get("fallback_models", ["deepseek-r1:8b"]),
            )
        except Exception:
            return []
        raw_stripped = str(raw or "").strip()
        if not raw_stripped or raw_stripped.upper().startswith("NONE"):
            return []
        lines = [ln.strip().lstrip("•-–*0123456789.) ") for ln in raw_stripped.splitlines()]
        followups = [ln for ln in lines if ln and len(ln) >= 4 and not ln.upper().startswith("NONE")][:3]
        return followups

    def _should_run_refined_second_pass(
        self,
        *,
        query: str,
        resolved_topic: str,
        seeds: list[dict[str, str]],
        sources: list[dict[str, Any]],
        crawled_pages: list[dict[str, Any]],
    ) -> bool:
        low = " ".join(str(query or "").split()).strip().lower()
        if not low:
            return False
        tier12 = sum(
            1
            for row in sources
            if str(row.get("source_tier", "tier3")).strip().lower() in {"tier1", "tier2"}
        )
        if resolved_topic == "combat_sports":
            wants_card_detail = any(
                token in low for token in ("full fight card", "card", "main card", "prelims", "bout order", "who is on")
            )
            return wants_card_detail and (len(sources) < 5 or len(crawled_pages) < 4 or tier12 < 2)
        if resolved_topic == "sports_event":
            wants_live_detail = any(
                token in low for token in ("lineup", "schedule", "odds", "spread", "moneyline", "kickoff", "tipoff", "broadcast")
            )
            return wants_live_detail and (len(sources) < 4 or len(crawled_pages) < 3 or tier12 < 2)
        if classify_fact_volatility(query, resolved_topic, query) == "volatile":
            return len(sources) < 3 and (len(seeds) < 6 or len(crawled_pages) < 2)
        return False

    def _domain_tier(self, host: str) -> tuple[str, float]:
        value = str(host or "").strip().lower()
        if not value:
            return "tier3", 0.45
        domain = value[4:] if value.startswith("www.") else value
        if self._domain_matches(domain, self.BANNED_SOURCES):
            return "banned_sources", 0.02
        if self._domain_matches(domain, self.TRUST_TIER_1):
            return "tier1", 1.0
        if self._domain_matches(domain, self.TRUST_TIER_2):
            return "tier2", 0.78
        if self._domain_matches(domain, self.TRUST_TIER_3):
            return "tier3", 0.52
        return "tier3", 0.45

    def _domain_matches(self, domain: str, candidates: set[str] | tuple[str, ...]) -> bool:
        root = str(domain or "").strip().lower()
        if not root:
            return False
        for candidate in candidates:
            item = str(candidate or "").strip().lower()
            if not item:
                continue
            if root == item or root.endswith("." + item):
                return True
        return False

    
    def _propaganda_penalty(self, title: str, snippet: str) -> float:
        text = f"{title} {snippet}".lower()
        hits = sum(1 for term in self.PROPAGANDA_TERMS if term in text)
        if hits == 0:
            return 0.0
        return min(0.25, hits * 0.05)

    def _low_signal_penalty(
        self,
        *,
        url: str,
        title: str,
        snippet: str,
        query_terms: set[str],
    ) -> tuple[float, list[str], bool]:
        text = f"{title} {snippet}".lower()
        url_low = str(url or "").lower()
        flags: list[str] = []
        penalty = 0.0

        support_intent_terms = {
            "unsupported",
            "browser",
            "compatibility",
            "javascript",
            "cookies",
            "captcha",
            "blocked",
            "denied",
            "forbidden",
            "access",
            "support",
        }
        support_intent = bool(query_terms & support_intent_terms)

        low_signal_hits = [term for term in self.LOW_SIGNAL_TEXT_TERMS if term in text]
        if low_signal_hits:
            penalty += min(0.52, 0.16 + (0.08 * min(len(low_signal_hits), 4)))
            flags.append("support_or_block_page")

        url_hits = [term for term in self.LOW_SIGNAL_URL_TERMS if term in url_low]
        if url_hits:
            penalty += min(0.22, 0.08 + (0.04 * min(len(url_hits), 3)))
            flags.append("blocked_url_pattern")

        nav_hits = sum(1 for term in self.NAVIGATION_NOISE_TERMS if term in text)
        if nav_hits >= 3:
            penalty += min(0.18, 0.05 + (0.025 * min(nav_hits, 5)))
            flags.append("navigation_heavy")

        if support_intent and ("support_or_block_page" in flags or "blocked_url_pattern" in flags):
            penalty = max(0.0, penalty - 0.24)
            flags.append("support_intent_query")

        quality_blocked = bool(
            ("support_or_block_page" in flags and "support_intent_query" not in flags)
            or ("blocked_url_pattern" in flags and "support_intent_query" not in flags and penalty >= 0.2)
        )
        return round(penalty, 3), flags, quality_blocked

    def _topic_domain_bonus(self, host: str, topic_type: str) -> float:
        topic = str(topic_type or "").strip().lower()
        domain = host[4:] if host.startswith("www.") else host
        if not domain:
            return 0.0
        if self._domain_matches(domain, self.BANNED_SOURCES):
            return -0.60

        # Coding/development runs should prefer exact primary documentation,
        # security advisories, release notes, package registries, and maintained
        # repos. Other topics still benefit from the same general source tiers.
        if topic == "technical":
            if self._domain_matches(domain, self.TRUST_TIER_1):
                return 0.08
            if self._domain_matches(domain, self.TRUST_TIER_2):
                return 0.04
            if self._domain_matches(domain, self.TRUST_TIER_3):
                return -0.04
            if any(tag in domain for tag in ("docs.", "developer.", "api.", "registry.", "packages.", "readthedocs")):
                return 0.03
            return 0.0

        if self._domain_matches(domain, self.TRUST_TIER_1):
            return 0.04
        if self._domain_matches(domain, self.TRUST_TIER_2):
            return 0.02
        if self._domain_matches(domain, self.TRUST_TIER_3):
            return -0.02
        return 0.0

    def _topic_signal_bonus(self, title: str, snippet: str, topic_type: str) -> float:
        text = f"{title} {snippet}".lower()
        topic = str(topic_type or "").strip().lower()
        if topic == "technical":
            terms = ("release notes", "changelog", "version", "api", "breaking change", "compatibility")
        elif topic == "finance":
            terms = ("earnings", "guidance", "revenue", "eps", "10-k", "10-q", "sec filing")
        elif topic == "current_events":
            terms = ("breaking", "live", "developing", "statement", "timeline", "update")
        elif topic == "law":
            terms = ("effective date", "statute", "bill", "court", "ruling", "regulation")
        elif topic == "education":
            terms = ("admission", "curriculum", "deadline", "accredited", "tuition", "syllabus")
        elif topic == "travel":
            terms = ("visa", "entry requirement", "travel advisory", "border", "flight", "departure")
        elif topic == "animal_care":
            terms = ("veterinary", "vaccination", "parasite", "pet food", "toxic", "animal welfare")
        elif topic == "food":
            terms = ("nutrition", "ingredient", "allergen", "food safety", "calorie", "recall")
        elif topic == "books":
            terms = ("edition", "isbn", "publisher", "publication date", "hardcover", "paperback")
        elif topic == "parenting":
            terms = ("pediatric", "milestone", "age range", "dosage", "safety", "guideline")
        elif topic == "business":
            terms = ("revenue", "margin", "guidance", "strategy", "market share", "executive")
        elif topic == "real_estate":
            terms = ("median price", "inventory", "mortgage", "cap rate", "vacancy", "housing starts")
        elif topic == "gaming":
            terms = ("patch notes", "season", "meta", "dlc", "release date", "esports")
        elif topic == "automotive":
            terms = ("msrp", "recall", "range", "mpg", "horsepower", "nhtsa")
        elif topic == "tv_shows":
            terms = ("season", "episode", "air date", "renewed", "cancelled", "showrunner")
        elif topic == "movies":
            terms = ("box office", "release date", "cast", "runtime", "director", "trailer")
        elif topic == "music":
            terms = ("album", "single", "tour", "release date", "chart", "label")
        elif topic == "art":
            terms = ("exhibition", "museum", "curator", "artist", "auction", "provenance")
        else:
            return 0.0
        hits = sum(1 for term in terms if term in text)
        if hits <= 0:
            return 0.0
        return min(0.08, hits * 0.02)

    _PRODUCT_INTENT_TERMS = frozenset({
        "price", "cost", "buy", "purchase", "review", "recommend", "best",
        "compare", "comparison", "worth", "vs", "versus", "alternative",
        "deal", "cheap", "expensive", "affordable", "rating", "ranked",
        "top", "worst", "avoid", "brand", "product", "model",
    })
    _COMMUNITY_REVIEW_DOMAINS = frozenset({
        "wirecutter.com", "thewirecutter.com", "consumerreports.org",
        "rtings.com", "reviewed.com", "pcmag.com",
    })

    def _product_community_bonus(self, host: str, query: str) -> float:
        """Boost community/review sources for product or pricing queries."""
        low_query = str(query or "").lower()
        if not any(t in low_query for t in self._PRODUCT_INTENT_TERMS):
            return 0.0
        domain = host[4:] if host.startswith("www.") else host
        if self._domain_matches(domain, {"reddit.com"}):
            return 0.08
        if self._domain_matches(domain, self._COMMUNITY_REVIEW_DOMAINS):
            return 0.06
        return 0.0

    _EMBED_MODEL = "qwen3-embedding:4b"

    def _embedding_filter_content(
        self,
        query: str,
        text: str,
        *,
        min_similarity: float = 0.30,
    ) -> str:
        """Filter scraped text paragraphs by embedding similarity to the query.

        Keeps only paragraphs with cosine similarity >= min_similarity.
        Falls back to the original text if embedding is unavailable or all chunks are filtered.
        """
        if not text or not query:
            return text
        # Split on double newlines (paragraph boundaries), fallback to 300-char chunks
        raw_chunks = [c.strip() for c in re.split(r"\n\n+", text) if c.strip()]
        if len(raw_chunks) <= 2:
            return text  # too short to bother filtering

        try:
            from shared_tools.ollama_client import OllamaClient
            client = OllamaClient()
            query_vec = client.embed(self._EMBED_MODEL, query[:800])
            if not query_vec:
                return text

            scored: list[tuple[float, str]] = []
            for chunk in raw_chunks:
                try:
                    chunk_vec = client.embed(self._EMBED_MODEL, chunk[:600])
                    if not chunk_vec or len(chunk_vec) != len(query_vec):
                        scored.append((0.0, chunk))
                        continue
                    dot = sum(a * b for a, b in zip(query_vec, chunk_vec))
                    import math as _math
                    na = _math.sqrt(sum(a * a for a in query_vec))
                    nb = _math.sqrt(sum(b * b for b in chunk_vec))
                    sim = dot / (na * nb) if na > 0 and nb > 0 else 0.0
                    scored.append((sim, chunk))
                except Exception:
                    scored.append((0.0, chunk))

            kept = [chunk for sim, chunk in scored if sim >= min_similarity]
            if not kept:
                # Keep top-3 by score as fallback rather than returning empty
                kept = [chunk for _, chunk in sorted(scored, key=lambda x: x[0], reverse=True)[:3]]
            return "\n\n".join(kept)
        except Exception:
            return text

    def _score_one_source(self, row: dict[str, Any], query_terms: set[str], query: str = "", topic_type: str = "general") -> dict[str, Any]:
        payload = dict(row)
        url_value = str(payload.get("url", "")).strip()
        host = self._hostname(url_value)
        tier_name, base_score = self._domain_tier(host)
        title = str(payload.get("title", "")).strip()
        snippet = str(payload.get("snippet", "")).strip()
        score = base_score
        payload.setdefault("retrieved_at", _now_iso())

        if url_value.lower().startswith("https://"):
            score += 0.03
        if len(snippet) >= 160:
            score += 0.05
        elif len(snippet) >= 80:
            score += 0.03

        hay = f"{title} {snippet}".lower()
        query_hit_count = 0
        if query_terms:
            query_hit_count = len([term for term in query_terms if term in hay])
            if query_hit_count > 0:
                score += min(0.12, query_hit_count * 0.02)
                if query_hit_count == 1 and len(query_terms) >= 4:
                    score -= 0.03
            else:
                score -= 0.08
        if str(payload.get("query_variant", "")).strip():
            score += 0.02

        score += self._topic_domain_bonus(host, topic_type)
        score += self._topic_signal_bonus(title, snippet, topic_type)
        score += self._product_community_bonus(host, query)
        score -= self._propaganda_penalty(title, snippet)
        low_signal_penalty, quality_flags, quality_blocked = self._low_signal_penalty(
            url=url_value,
            title=title,
            snippet=snippet,
            query_terms=query_terms,
        )
        score -= low_signal_penalty
        banned_source = tier_name == "banned_sources"
        if banned_source:
            quality_flags = list(quality_flags) + ["banned_source_domain"]
            quality_blocked = True
        payload["source_domain"] = host
        payload["source_tier"] = tier_name
        payload = enrich_source_metadata(payload, query=query, topic_type=topic_type)
        if not banned_source:
            score += float(payload.get("freshness_score", 0.0)) * 0.08
            score += float(payload.get("volatility_fit_score", 0.0)) * 0.05
            if bool(payload.get("stale_for_query", False)):
                score -= 0.08
            score += self._domain_rep.get_adjustment(host)
            score = max(0.05, min(1.0, round(score, 3)))
        else:
            score = 0.02
        payload["source_score"] = score
        payload["query_term_hits"] = int(query_hit_count)
        payload["quality_penalty"] = float(low_signal_penalty)
        payload["quality_flags"] = quality_flags
        payload["quality_blocked"] = bool(quality_blocked)
        return payload

    def _apply_source_scoring(
        self,
        sources: list[dict[str, Any]],
        query: str,
        enabled: bool,
        topic_type: str = "general",
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        resolved_topic = self._resolve_topic_type(query, topic_type)
        if not sources:
            return [], {
                "enabled": bool(enabled),
                "applied": False,
                "strategy": "developer_domain_tiers_v1",
                "topic_type": resolved_topic,
                "tier_counts": {"tier1": 0, "tier2": 0, "tier3": 0, "banned_sources": 0},
                "top_score": 0.0,
            }

        terms = self._query_terms(query)
        scored = [self._score_one_source(row, terms, query=query, topic_type=resolved_topic) for row in sources]
        if bool(enabled):
            volatility = classify_fact_volatility(query, resolved_topic)
            if volatility == "volatile":
                # For volatile topics (live events, breaking news, prices), freshness
                # must dominate — a fresh tier-2 source beats a stale tier-1 source.
                scored.sort(
                    key=lambda x: (
                        0 if bool(x.get("stale_for_query", False)) else 1,
                        float(x.get("freshness_score", 0.0)),
                        float(x.get("source_score", 0.0)),
                        len(str(x.get("snippet", ""))),
                    ),
                    reverse=True,
                )
            else:
                # Stable and semi-volatile: authority (source_score) primary, freshness tiebreaker.
                scored.sort(
                    key=lambda x: (
                        float(x.get("source_score", 0.0)),
                        float(x.get("freshness_score", 0.0)),
                        -float(x.get("source_age_hours", 0.0) or 0.0),
                        len(str(x.get("snippet", ""))),
                    ),
                    reverse=True,
                )

        tier_counts = {"tier1": 0, "tier2": 0, "tier3": 0, "banned_sources": 0}
        for row in scored:
            tier = str(row.get("source_tier", "tier3"))
            if tier not in tier_counts:
                tier = "tier3"
            tier_counts[tier] += 1

        freshness = [float(r.get("freshness_score", 0.0)) for r in scored]
        summary = {
            "enabled": bool(enabled),
            "applied": bool(enabled),
            "strategy": "developer_domain_tiers_v1_freshness",
            "topic_type": resolved_topic,
            "tier_counts": tier_counts,
            "top_score": float(scored[0].get("source_score", 0.0)),
            "avg_freshness": round(sum(freshness) / len(freshness), 3) if freshness else 0.0,
            "stale_count": sum(1 for r in scored if bool(r.get("stale_for_query", False))),
        }
        return scored, summary

    def _normalize_money_to_usd(self, raw: str) -> str | None:
        text = str(raw or "").strip().lower()
        if not text:
            return None
        text = text.replace(",", "").replace("$", "").strip()
        factor = 1.0
        if text.endswith("billion"):
            factor = 1_000_000_000.0
            text = text[:-7].strip()
        elif text.endswith("million"):
            factor = 1_000_000.0
            text = text[:-7].strip()
        elif text.endswith("thousand"):
            factor = 1_000.0
            text = text[:-8].strip()
        elif text.endswith("b"):
            factor = 1_000_000_000.0
            text = text[:-1].strip()
        elif text.endswith("m"):
            factor = 1_000_000.0
            text = text[:-1].strip()
        elif text.endswith("k"):
            factor = 1_000.0
            text = text[:-1].strip()
        text = re.sub(r"[^0-9.]+", "", text)
        if not text:
            return None
        try:
            value = float(text) * factor
        except Exception:
            return None
        if value <= 0:
            return None
        return f"${int(round(value)):,}"

    def _normalize_isbn(self, raw: str) -> str | None:
        token = re.sub(r"[^0-9Xx]", "", str(raw or ""))
        if len(token) == 10 and re.fullmatch(r"\d{9}[0-9Xx]", token):
            return token.upper()
        if len(token) == 13 and token.isdigit() and token.startswith(("978", "979")):
            return token
        return None

    def _extract_conflict_values(self, row: dict[str, Any], topic_type: str = "general") -> dict[str, set[str]]:
        title = str(row.get("title", "")).strip()
        snippet = str(row.get("snippet", "")).strip()
        text = f"{title}\n{snippet}"
        low = text.lower()
        topic = str(topic_type or "").strip().lower()

        date_values = {
            v.strip()
            for v in re.findall(
                r"\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\s+\d{1,2},?\s+(?:19|20)\d{2}\b",
                low,
                flags=re.IGNORECASE,
            )
        }
        year_values = {
            v.strip()
            for v in re.findall(r"\b(?:19|20)\d{2}\b", low)
        }
        number_values = {
            v.strip()
            for v in re.findall(r"\b\d{1,4}(?:\.\d+)?\b", low)
        }
        number_values = {v for v in number_values if v not in year_values}

        matchup_values: set[str] = set()
        for left, right in re.findall(
            r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\s+vs\.?\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\b",
            text,
        ):
            matchup_values.add(f"{left.strip()} vs {right.strip()}".lower())

        isbn_values: set[str] = set()
        msrp_values: set[str] = set()
        range_values: set[str] = set()
        mpg_values: set[str] = set()
        runtime_values: set[str] = set()
        box_office_values: set[str] = set()

        if topic == "books":
            for chunk in re.findall(r"\bISBN(?:-1[03])?:?\s*([0-9Xx\- ]{10,20})\b", text, flags=re.IGNORECASE):
                normalized = self._normalize_isbn(chunk)
                if normalized:
                    isbn_values.add(normalized)
            for chunk in re.findall(r"\b97[89][0-9\- ]{10,20}\b", text):
                normalized = self._normalize_isbn(chunk)
                if normalized:
                    isbn_values.add(normalized)

        if topic == "automotive":
            money_chunks = re.findall(
                r"(?:msrp|starting at|starts at|price(?:d)? at)\s*[:\-]?\s*(\$?\s*\d[\d,]*(?:\.\d+)?(?:\s*[kmb]|(?:\s*(?:million|billion|thousand)))?)",
                low,
                flags=re.IGNORECASE,
            )
            for chunk in money_chunks:
                normalized = self._normalize_money_to_usd(chunk)
                if normalized:
                    msrp_values.add(normalized)
            for miles in re.findall(
                r"\b(?:epa\s+)?(?:range|estimated range|driving range)?\s*(?:of|:|is|up to)?\s*(\d{2,4})\s*(?:mile|miles|mi)\b",
                low,
                flags=re.IGNORECASE,
            ):
                range_values.add(f"{int(miles)} mi")
            for miles in re.findall(r"\b(\d{2,4})\s*-\s*mile\s+range\b", low, flags=re.IGNORECASE):
                range_values.add(f"{int(miles)} mi")
            for city, hwy in re.findall(r"\b(\d{1,3})\s*/\s*(\d{1,3})\s*mpg\b", low, flags=re.IGNORECASE):
                mpg_values.add(f"{int(city)}/{int(hwy)} mpg")
            for mpg in re.findall(r"\b(\d{1,3})\s*mpg\b", low, flags=re.IGNORECASE):
                mpg_values.add(f"{int(mpg)} mpg")

        if topic == "movies":
            for chunk in re.findall(
                r"\b(?:box office|gross|opening weekend|opening)\b[^$]{0,40}(\$?\s*\d[\d,]*(?:\.\d+)?(?:\s*[kmb]|(?:\s*(?:million|billion|thousand)))?)",
                low,
                flags=re.IGNORECASE,
            ):
                normalized = self._normalize_money_to_usd(chunk)
                if normalized:
                    box_office_values.add(normalized)
            for mins in re.findall(
                r"\b(?:runtime|run time|running time)\s*(?:of|:)?\s*(\d{2,3})\s*(?:min|mins|minutes)\b",
                low,
                flags=re.IGNORECASE,
            ):
                runtime_values.add(f"{int(mins)} min")
            if not runtime_values:
                for mins in re.findall(r"\b(\d{2,3})\s*(?:min|mins|minutes)\b", low, flags=re.IGNORECASE):
                    runtime_values.add(f"{int(mins)} min")

        return {
            "date": date_values,
            "year": year_values,
            "number": number_values,
            "matchup": matchup_values,
            "isbn": isbn_values,
            "msrp": msrp_values,
            "range": range_values,
            "mpg": mpg_values,
            "runtime": runtime_values,
            "box_office": box_office_values,
        }

    def _detect_source_conflicts(
        self,
        sources: list[dict[str, Any]],
        query: str,
        enabled: bool,
        topic_type: str = "general",
    ) -> dict[str, Any]:
        resolved_topic = self._resolve_topic_type(query, topic_type)
        summary = {
            "enabled": bool(enabled),
            "applied": bool(enabled),
            "conflict_count": 0,
            "conflicts": [],
            "note": "",
            "topic_type": resolved_topic,
        }
        if not sources or not bool(enabled):
            return summary

        low_query = str(query or "").lower()
        date_intent = any(
            token in low_query
            for token in (
                "when",
                "date",
                "year",
                "timeline",
                "schedule",
                "release date",
                "effective date",
                "deadline",
                "air date",
                "today",
                "tonight",
                "this week",
                "this month",
                "this year",
            )
        )
        numeric_intent = any(
            token in low_query
            for token in (
                "how many",
                "price",
                "cost",
                "score",
                "record",
                "rank",
                "percent",
            )
        )
        matchup_intent = any(token in low_query for token in (" vs ", " versus ", "matchup", "head to head"))

        topic_date_intent_tokens: dict[str, tuple[str, ...]] = {
            "law": ("effective date", "ruling date", "filing date"),
            "education": ("deadline", "enrollment date"),
            "travel": ("departure", "arrival", "travel advisory"),
            "books": ("publication date", "release date"),
            "gaming": ("patch date", "release date", "season start"),
            "tv_shows": ("air date", "episode date", "season release"),
            "movies": ("release date", "premiere date"),
            "music": ("release date", "tour date"),
            "art": ("exhibition date", "auction date"),
        }
        topic_numeric_intent_tokens: dict[str, tuple[str, ...]] = {
            "education": ("tuition", "credit hours", "acceptance rate"),
            "food": ("calories", "grams", "serving size"),
            "books": ("isbn", "page count"),
            "parenting": ("age", "months", "years", "dosage"),
            "business": ("revenue", "margin", "market share", "quarter"),
            "real_estate": ("mortgage rate", "median price", "inventory", "cap rate", "rent"),
            "gaming": ("player count", "meta rank"),
            "automotive": ("msrp", "range", "mpg", "horsepower"),
            "tv_shows": ("runtime", "rating"),
            "movies": ("box office", "runtime", "budget", "rating"),
            "music": ("chart", "streams"),
            "art": ("auction price", "estimate"),
        }
        topic_matchup_tokens: dict[str, tuple[str, ...]] = {
            "combat_sports": ("fight", "bout", "matchup", "main event", "vs"),
            "sports_event": ("game", "matchup", "vs", "head to head"),
        }
        if any(token in low_query for token in topic_date_intent_tokens.get(resolved_topic, ())):
            date_intent = True
        if any(token in low_query for token in topic_numeric_intent_tokens.get(resolved_topic, ())):
            numeric_intent = True
        if any(token in low_query for token in topic_matchup_tokens.get(resolved_topic, ())):
            matchup_intent = True

        if not (date_intent or numeric_intent or matchup_intent):
            return summary

        buckets: dict[str, dict[str, set[int]]] = {
            "date": {},
            "year": {},
            "number": {},
            "matchup": {},
            "isbn": {},
            "msrp": {},
            "range": {},
            "mpg": {},
            "runtime": {},
            "box_office": {},
        }
        for idx, row in enumerate(sources, start=1):
            values = self._extract_conflict_values(row, topic_type=resolved_topic)
            for key, entries in values.items():
                for entry in entries:
                    if not entry:
                        continue
                    buckets[key].setdefault(entry, set()).add(idx)

        conflict_order: list[str] = list(
            {
                "books": ("isbn", "date", "year", "number"),
                "automotive": ("msrp", "range", "mpg", "date", "year", "number"),
                "movies": ("box_office", "runtime", "date", "year", "number"),
            }.get(resolved_topic, ("date", "year", "matchup", "number"))
        )
        if not date_intent:
            conflict_order = [key for key in conflict_order if key not in {"date", "year"}]
        if not matchup_intent:
            conflict_order = [key for key in conflict_order if key != "matchup"]
        if not numeric_intent:
            conflict_order = [key for key in conflict_order if key != "number"]
        if not conflict_order:
            return summary

        conflicts: list[dict[str, Any]] = []
        for key in conflict_order:
            candidates = [
                {"value": value, "sources": sorted(list(indexes))}
                for value, indexes in buckets[key].items()
                if indexes
            ]
            if len(candidates) < 2:
                continue
            candidates.sort(key=lambda x: (len(x["sources"]), x["value"]), reverse=True)
            top_values = candidates[:4]
            source_union = sorted({src for row in top_values for src in row["sources"]})
            if len(source_union) < 2:
                continue
            conflicts.append(
                {
                    "type": key,
                    "values": top_values,
                    "source_coverage": len(source_union),
                }
            )

        summary["conflicts"] = conflicts
        summary["conflict_count"] = len(conflicts)
        summary["topic_type"] = resolved_topic
        if conflicts:
            parts = []
            for row in conflicts[:3]:
                names = [str(x.get("value", "")) for x in row.get("values", [])[:2]]
                key_label = str(row.get("type", "unknown")).replace("_", " ")
                parts.append(f"{key_label}: {' vs '.join(names)}")
            summary["note"] = "Potential source conflicts detected: " + "; ".join(parts)
        return summary

    def search(self, query: str, max_results: int | None = None) -> list[dict[str, str]]:
        settings = self._load_settings()
        k = max_results if max_results is not None else int(settings.get("max_results", 8))
        limit = max(1, min(int(k), 20))
        retry_attempts = max(1, int(settings.get("search_retry_attempts", 3)))
        query_text = query.strip()
        if not query_text:
            return []
        provider = str(settings.get("provider", "auto")).strip().lower()
        searx_rows: list[dict[str, str]] = []
        ddg_html_rows: list[dict[str, str]] = []
        ddg_api_rows: list[dict[str, str]] = []

        if provider in {"auto", "searxng"}:
            now = time.time()
            if now >= self._searxng_backoff_until:
                for _ in range(retry_attempts):
                    try:
                        searx_rows = self._search_searxng(query_text, limit, settings)
                    except Exception:
                        searx_rows = []
                    if searx_rows:
                        break
                if not searx_rows:
                    self._searxng_backoff_until = max(self._searxng_backoff_until, time.time() + 120.0)
            if provider == "searxng":
                return searx_rows[:limit]

        if provider in {"auto", "duckduckgo_html"}:
            for _ in range(retry_attempts):
                try:
                    ddg_html_rows = self._search_duckduckgo_html(query_text, limit)
                except Exception:
                    ddg_html_rows = []
                if ddg_html_rows:
                    break
            if provider == "duckduckgo_html":
                return ddg_html_rows[:limit]

        if provider in {"auto", "duckduckgo_api"}:
            for _ in range(retry_attempts):
                try:
                    ddg_api_rows = self._search_duckduckgo_api(query_text, limit)
                except Exception:
                    ddg_api_rows = []
                if ddg_api_rows:
                    break
            if provider == "duckduckgo_api":
                return ddg_api_rows[:limit]

        merged = self._merge_results(searx_rows, ddg_html_rows, limit)
        merged = self._merge_results(merged, ddg_api_rows, limit)
        return merged[:limit]

    def _hostname(self, url: str) -> str:
        return str(urllib.parse.urlsplit(url).hostname or "").strip().lower()

    def _normalize_url(self, raw_url: str, base_url: str = "") -> str:
        text = str(raw_url or "").strip()
        if not text:
            return ""
        joined = urllib.parse.urljoin(base_url, text) if base_url else text
        parsed = urllib.parse.urlsplit(joined)
        if parsed.scheme not in {"http", "https"}:
            return ""
        if not parsed.netloc:
            return ""
        # Drop fragments, normalize path slashes.
        path = re.sub(r"/{2,}", "/", parsed.path or "/")
        normalized = urllib.parse.urlunsplit((parsed.scheme, parsed.netloc.lower(), path, parsed.query, ""))
        return normalized

    def _can_crawl_url(self, url: str) -> bool:
        low = url.lower()
        blocked_ext = (
            ".pdf",
            ".jpg",
            ".jpeg",
            ".png",
            ".gif",
            ".webp",
            ".svg",
            ".zip",
            ".rar",
            ".7z",
            ".exe",
            ".dmg",
            ".mp3",
            ".mp4",
            ".mov",
            ".avi",
        )
        return not low.endswith(blocked_ext)

    def _extract_urls_from_text(self, text: str, limit: int = 18) -> list[str]:
        rows: list[str] = []
        seen: set[str] = set()
        for match in re.finditer(r"https?://[^\s)\]}>\"']+", str(text or ""), flags=re.IGNORECASE):
            candidate = self._normalize_url(match.group(0))
            if not candidate or candidate in seen:
                continue
            seen.add(candidate)
            rows.append(candidate)
            if len(rows) >= max(1, limit):
                break
        return rows

    def _fetch_page_basic(self, url: str, timeout_sec: int, text_chars: int, retry_attempts: int) -> dict[str, Any]:
        req = urllib.request.Request(
            url=url,
            headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) FoxforgeCrawler/1.0",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            },
            method="GET",
        )
        last_exc = None
        for _ in range(max(1, retry_attempts)):
            try:
                ctx = self._urlopen(req, timeout=int(timeout_sec))
                with ctx as resp:
                    ctype = str(resp.headers.get("Content-Type", "")).lower()
                    if "text/html" not in ctype and "application/xhtml+xml" not in ctype:
                        raise RuntimeError(f"unsupported content type: {ctype or 'unknown'}")
                    raw = resp.read(2_000_000)
                break
            except Exception as exc:
                last_exc = exc
                raw = b""
                continue
        if not raw:
            raise RuntimeError(str(last_exc or "fetch failed"))
        body = raw.decode("utf-8", errors="ignore")
        extractor = _PageExtractor()
        extractor.feed(body)
        title = extractor.title() or url
        snippet = extractor.snippet(max_chars=text_chars)
        return {
            "url": url,
            "title": title,
            "snippet": snippet,
            "links": extractor.links,
        }

    # ------------------------------------------------------------------
    # Web chunk cache — DB storage / retrieval / purge
    # ------------------------------------------------------------------

    def _db_connect(self) -> sqlite3.Connection:
        from shared_tools.migrations import initialize_database
        initialize_database(self.repo_root)
        return _db_connect_shared(self.repo_root)

    def _store_web_chunks(self, project: str, sources: list[dict[str, Any]], ttl_days: int = 14) -> None:
        """Persist crawled source snippets into web_cache_chunks for long-term retrieval."""
        if not sources:
            return
        now = datetime.now(timezone.utc)
        expires = (now + timedelta(days=ttl_days)).isoformat()
        crawled_at = now.isoformat()
        try:
            conn = self._db_connect()
            with conn:
                for src in sources:
                    url = str(src.get("url", "")).strip()
                    snippet = str(src.get("snippet", "")).strip()
                    if not url or not snippet:
                        continue
                    chunk_id = hashlib.sha256(f"{project}:{url}".encode()).hexdigest()[:32]
                    domain = str(urllib.parse.urlsplit(url).hostname or "").lower().removeprefix("www.")
                    conn.execute(
                        """
                        INSERT INTO web_cache_chunks
                            (id, project, url, title, domain, snippet, source_score, source_tier, crawled_at, expires_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        ON CONFLICT(id) DO UPDATE SET
                            snippet=excluded.snippet,
                            title=excluded.title,
                            source_score=excluded.source_score,
                            source_tier=excluded.source_tier,
                            crawled_at=excluded.crawled_at,
                            expires_at=excluded.expires_at
                        """,
                        (
                            chunk_id,
                            project,
                            url,
                            str(src.get("title", "")).strip(),
                            domain,
                            snippet,
                            float(src.get("source_score", 0.0)),
                            str(src.get("source_tier", "tier3")),
                            crawled_at,
                            expires,
                        ),
                    )
        except Exception:
            pass

    def _purge_expired_web_chunks(self) -> int:
        """Delete expired rows from web_cache_chunks. Returns count deleted."""
        try:
            conn = self._db_connect()
            now_iso = datetime.now(timezone.utc).isoformat()
            with conn:
                cur = conn.execute(
                    "DELETE FROM web_cache_chunks WHERE expires_at < ?", (now_iso,)
                )
                return cur.rowcount or 0
        except Exception:
            return 0

    def _purge_old_source_log(self, retain_days: int = 30) -> None:
        """Remove entries older than retain_days from sources.jsonl."""
        if not self.sources_log_path.exists():
            return
        cutoff = (datetime.now(timezone.utc) - timedelta(days=retain_days)).isoformat()
        try:
            lines = self.sources_log_path.read_text(encoding="utf-8", errors="ignore").splitlines()
            kept = []
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                try:
                    ts = json.loads(line).get("ts", "")
                    if str(ts) >= cutoff:
                        kept.append(line)
                except Exception:
                    kept.append(line)  # keep unparseable lines
            self.sources_log_path.write_text("\n".join(kept) + ("\n" if kept else ""), encoding="utf-8")
        except Exception:
            pass

    def _query_cached_chunks(self, project: str, limit: int = 8) -> list[dict[str, Any]]:
        """Retrieve non-expired web chunks for a project from the DB cache."""
        try:
            conn = self._db_connect()
            now_iso = datetime.now(timezone.utc).isoformat()
            rows = conn.execute(
                """
                SELECT url, title, domain, snippet, source_score, source_tier, crawled_at
                FROM web_cache_chunks
                WHERE project = ? AND expires_at > ?
                ORDER BY crawled_at DESC
                LIMIT ?
                """,
                (project, now_iso, limit),
            ).fetchall()
            return [dict(r) for r in rows]
        except Exception:
            return []

    # ------------------------------------------------------------------
    # Wikipedia article fetcher
    # ------------------------------------------------------------------

    _SPORTS_TOPIC_TYPES: frozenset[str] = frozenset({
        "sports", "combat_sports", "sports_event", "current_events",
    })

    def _fetch_wikipedia_extract(self, query: str, text_chars: int = 5000, topic_type: str = "general") -> dict[str, Any] | None:
        """Fetch the best-matching Wikipedia article for a query via the MediaWiki API.

        For sports/current_events topics, searches with today's full date first so
        tonight's event ranks above the next scheduled one. Falls back through
        progressively wider date windows until a result is found.
        """
        base_query = " ".join(str(query or "").split()).strip()
        if not base_query:
            return None

        now = datetime.now(timezone.utc)
        today_full = now.strftime("%B %-d %Y")    # "April 11 2026"
        today_month = now.strftime("%B %Y")        # "April 2026"
        current_year = str(now.year)              # "2026"

        is_sports = str(topic_type).lower() in self._SPORTS_TOPIC_TYPES

        # Build candidate search queries ordered from most-specific to least
        if is_sports:
            search_candidates = [
                f"{base_query} {today_full}",
                f"{base_query} {today_month}",
                f"{base_query} {current_year}",
                base_query,
            ]
        else:
            search_candidates = [
                f"{base_query} {current_year}",
                base_query,
            ]

        def _wiki_search(q: str) -> list[dict[str, Any]]:
            url = (
                "https://en.wikipedia.org/w/api.php"
                f"?action=query&list=search&srsearch={urllib.parse.quote(q)}"
                "&format=json&utf8=1&srlimit=5"
            )
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "Foxforge/1.0"})
                with self._urlopen(req, timeout=10) as resp:
                    return json.loads(resp.read().decode("utf-8", errors="ignore")).get("query", {}).get("search", [])
            except Exception:
                return []

        def _score_hit(hit: dict[str, Any], prefer_date: str) -> float:
            """Prefer hits whose snippet/title contains today's date string."""
            combined = (str(hit.get("title", "")) + " " + str(hit.get("snippet", ""))).lower()
            prefer_low = prefer_date.lower()
            return 1.0 if prefer_low in combined else 0.0

        hits: list[dict[str, Any]] = []
        chosen_date_hint = today_full if is_sports else current_year

        for candidate in search_candidates:
            results = _wiki_search(candidate)
            if not results:
                continue
            if is_sports and candidate.endswith(today_full):
                # Score results by date proximity — pick the one mentioning today
                scored = sorted(results, key=lambda h: _score_hit(h, today_full), reverse=True)
                hits = scored
            else:
                hits = results
            break

        if not hits:
            return None

        page_id = int(hits[0].get("pageid", 0))
        page_title = str(hits[0].get("title", "")).strip()
        if not page_id or not page_title:
            return None

        # Fetch the full article extract (not just intro) via the extracts prop
        extract_url = (
            "https://en.wikipedia.org/w/api.php"
            f"?action=query&pageids={page_id}"
            "&prop=extracts&exintro=0&explaintext=1"
            "&format=json&utf8=1"
        )
        try:
            req3 = urllib.request.Request(
                url=extract_url,
                headers={"User-Agent": "Foxforge/1.0 (research assistant; contact: local)"},
            )
            with self._urlopen(req3, timeout=12) as resp3:
                extract_data = json.loads(resp3.read().decode("utf-8", errors="ignore"))
        except Exception:
            return None

        pages = extract_data.get("query", {}).get("pages", {})
        page = pages.get(str(page_id), {})
        extract = str(page.get("extract", "")).strip()
        if not extract:
            return None

        # Trim to text_chars, prefer sentence boundaries
        if len(extract) > text_chars:
            cut = extract[:text_chars].rsplit(".", 1)[0].strip()
            extract = (cut or extract[:text_chars]).strip() + "..."

        wiki_url = f"https://en.wikipedia.org/wiki/{urllib.parse.quote(page_title.replace(' ', '_'))}"
        return {
            "url": wiki_url,
            "title": page_title,
            "snippet": extract,
            "depth": 0,
            "_wikipedia": True,
        }

    # ── Reddit ────────────────────────────────────────────────────────────────

    def _is_reddit_url(self, url: str) -> bool:
        host = self._hostname(str(url or "")).lower()
        return host in {"reddit.com", "www.reddit.com", "old.reddit.com", "new.reddit.com"}

    def _fetch_reddit_json(self, url: str, text_chars: int = 2500) -> dict[str, Any] | None:
        """Fetch a Reddit thread via the public JSON API and return clean text content."""
        if not url:
            return None
        # Respect backoff
        if time.time() < self._reddit_backoff_until:
            return None

        # Normalize to old.reddit.com for consistent JSON responses
        normalized = re.sub(r"https?://(www\.|new\.)?reddit\.com", "https://old.reddit.com", url.rstrip("/"))
        # Remove trailing query strings for cleaner JSON endpoint
        normalized = normalized.split("?")[0]
        json_url = normalized + "/.json?limit=25"

        try:
            req = urllib.request.Request(
                json_url,
                headers={
                    "User-Agent": "Foxforge/1.0 (research assistant)",
                    "Accept": "application/json",
                },
            )
            with self._urlopen(req, timeout=10) as resp:
                raw = resp.read()
        except Exception:
            self._reddit_backoff_until = time.time() + 30.0
            return None

        try:
            data = json.loads(raw.decode("utf-8", errors="ignore"))
        except Exception:
            return None

        # Rate-limit check via headers is not accessible here; use heuristic
        if isinstance(data, dict) and data.get("error"):
            self._reddit_backoff_until = time.time() + 60.0
            return None

        parts: list[str] = []
        title = ""

        try:
            # Reddit JSON for a thread returns a 2-element list: [post_listing, comments_listing]
            if isinstance(data, list) and len(data) >= 1:
                post_data = data[0].get("data", {}).get("children", [{}])[0].get("data", {})
                title = str(post_data.get("title", "")).strip()
                selftext = str(post_data.get("selftext", "")).strip()
                if title:
                    parts.append(title)
                if selftext and selftext != "[deleted]" and selftext != "[removed]":
                    parts.append(selftext[:1500])

                # Extract top comments from second listing
                if len(data) >= 2:
                    comments = data[1].get("data", {}).get("children", [])
                    comment_texts: list[tuple[int, str]] = []
                    for child in comments:
                        c = child.get("data", {})
                        body = str(c.get("body", "")).strip()
                        score = int(c.get("score", 0))
                        if body and body not in ("[deleted]", "[removed]") and score > 0:
                            comment_texts.append((score, body[:600]))
                    # Sort by score descending, take top 8
                    comment_texts.sort(reverse=True)
                    for _, body in comment_texts[:8]:
                        parts.append(body)
            elif isinstance(data, dict):
                # Search results listing
                for child in data.get("data", {}).get("children", []):
                    post = child.get("data", {})
                    t = str(post.get("title", "")).strip()
                    s = str(post.get("selftext", "")).strip()
                    if t:
                        parts.append(t)
                    if s and s not in ("[deleted]", "[removed]"):
                        parts.append(s[:400])
        except Exception:
            return None

        if not parts:
            return None

        combined = "\n\n".join(parts)
        if len(combined) > text_chars:
            combined = combined[:text_chars].rsplit(".", 1)[0].strip() + "..."

        return {
            "url": url,
            "title": title or "Reddit thread",
            "snippet": combined,
            "depth": 0,
            "_reddit": True,
        }

    def _reddit_search(self, query: str, limit: int = 5) -> list[dict[str, str]]:
        """Search Reddit's public JSON search API and return seed results."""
        if time.time() < self._reddit_backoff_until:
            return []
        base_query = " ".join(str(query or "").split()).strip()
        if not base_query:
            return []
        search_url = (
            "https://www.reddit.com/search.json"
            f"?q={urllib.parse.quote(base_query)}&sort=relevance&limit={min(limit, 10)}&type=link"
        )
        try:
            req = urllib.request.Request(
                search_url,
                headers={
                    "User-Agent": "Foxforge/1.0 (research assistant)",
                    "Accept": "application/json",
                },
            )
            with self._urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode("utf-8", errors="ignore"))
        except Exception:
            self._reddit_backoff_until = time.time() + 30.0
            return []

        results: list[dict[str, str]] = []
        try:
            for child in data.get("data", {}).get("children", []):
                post = child.get("data", {})
                permalink = str(post.get("permalink", "")).strip()
                if not permalink:
                    continue
                full_url = f"https://www.reddit.com{permalink}"
                title = str(post.get("title", "")).strip()
                selftext = str(post.get("selftext", "")).strip()
                snippet = selftext[:300] if selftext and selftext not in ("[deleted]", "[removed]") else title
                results.append({
                    "url": full_url,
                    "title": title,
                    "snippet": snippet,
                })
        except Exception:
            pass
        return results

    def _is_product_query(self, query: str) -> bool:
        """Return True if the query appears to be about products, prices, or reviews."""
        low = str(query or "").lower()
        return any(t in low for t in self._PRODUCT_INTENT_TERMS)

    def _fetch_page_crawl4ai(self, url: str, settings: dict[str, Any], text_chars: int) -> dict[str, Any]:
        if not bool(settings.get("crawl4ai_enabled", True)):
            raise RuntimeError("crawl4ai disabled")

        base_url = str(settings.get("crawl4ai_base_url", "http://127.0.0.1:11235")).strip().rstrip("/")
        if not base_url:
            raise RuntimeError("crawl4ai base URL is empty")

        timeout_sec = int(settings.get("crawl4ai_timeout_sec", 40))
        retry_attempts = int(settings.get("crawl4ai_retry_attempts", 2))
        css_selector = str(settings.get("crawl4ai_css_selector", "")).strip()

        payload = {
            "urls": [url],
            "bypass_cache": True,
            # Drop nav/header/footer/aside before markdown generation to reduce JS boilerplate.
            "excluded_tags": ["nav", "header", "footer", "aside", "script", "style", "noscript"],
            # Discard content blocks under 10 words — kills one-liner link menus and button text.
            "word_count_threshold": 10,
            # Auto-remove cookie banners, modals, and overlay elements.
            "remove_overlay_elements": True,
        }
        if css_selector:
            payload["css_selector"] = css_selector

        req = urllib.request.Request(
            url=f"{base_url}/crawl",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "User-Agent": "Foxforge/1.0",
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
            method="POST",
        )

        last_exc: Exception | None = None
        raw_payload: dict[str, Any] | None = None
        for _ in range(max(1, retry_attempts)):
            try:
                with self._urlopen(req, timeout=timeout_sec) as resp:
                    raw_payload = json.loads(resp.read().decode("utf-8", errors="ignore"))
                break
            except Exception as exc:
                last_exc = exc
                raw_payload = None
                continue

        if not isinstance(raw_payload, dict):
            raise RuntimeError(str(last_exc or "crawl4ai response missing"))

        rows: Any = raw_payload.get("results")
        if not isinstance(rows, list):
            rows = raw_payload.get("data")
        if not isinstance(rows, list):
            single = raw_payload.get("result")
            if isinstance(single, dict):
                rows = [single]
        if not isinstance(rows, list) or not rows:
            raise RuntimeError("crawl4ai returned no crawl rows")

        first = rows[0] if isinstance(rows[0], dict) else {}

        # crawl4ai >= 0.5 returns markdown as a dict: {raw_markdown, fit_markdown, ...}
        # Older versions returned it as a plain string. Handle both.
        _md_raw = first.get("markdown", "")
        if isinstance(_md_raw, dict):
            markdown = str(_md_raw.get("fit_markdown", "") or _md_raw.get("raw_markdown", "")).strip()
            _md_for_links = markdown
        else:
            markdown = str(_md_raw).strip()
            _md_for_links = markdown

        text = str(first.get("text", "")).strip()

        # Title: prefer direct field, fall back to metadata dict (crawl4ai >= 0.5).
        title = str(first.get("title", "")).strip()
        if not title:
            _meta = first.get("metadata", {})
            if isinstance(_meta, dict):
                title = str(_meta.get("title", "")).strip()

        snippet_source = markdown or text
        if not snippet_source:
            # Last fallback for unknown response schemas.
            snippet_source = json.dumps(first, ensure_ascii=True)
        snippet_source = _clean_crawl4ai_markdown(snippet_source)
        snippet = " ".join(snippet_source.split())
        if len(snippet) > text_chars:
            cut = snippet[:text_chars].rsplit(" ", 1)[0].strip()
            snippet = (cut or snippet[:text_chars]).strip() + "..."
        # Extract links for depth crawling.
        # Priority 1: crawl4ai's own structured links dict (hrefs already resolved).
        # Priority 2: parse markdown for both absolute and relative links — relative
        #             paths like "/path/to/page" are resolved by the caller via
        #             _normalize_url(href, base_url=current_url) in _crawl_sources.
        # NOTE: _extract_urls_from_text only matches https?:// absolute URLs and
        # misses the relative links that dominate crawl4ai markdown output.
        links: list[str] = []
        _seen_links: set[str] = set()
        _limit = 28

        _c4ai_links = first.get("links")
        if isinstance(_c4ai_links, dict):
            for _link_list in (_c4ai_links.get("internal", []), _c4ai_links.get("external", [])):
                if not isinstance(_link_list, list):
                    continue
                for _lobj in _link_list:
                    if len(links) >= _limit:
                        break
                    href = (
                        str(_lobj.get("href", "")).strip()
                        if isinstance(_lobj, dict)
                        else str(_lobj).strip()
                    )
                    if not href or href in _seen_links:
                        continue
                    _seen_links.add(href)
                    links.append(href)
                if len(links) >= _limit:
                    break

        if len(links) < _limit:
            _md_text = _md_for_links or text
            for _m in re.finditer(r'\]\(((?:/|https?://)[^\s)\"\'<>]+)\)', _md_text):
                if len(links) >= _limit:
                    break
                href = _m.group(1).strip()
                if not href or href in _seen_links:
                    continue
                _seen_links.add(href)
                links.append(href)

        return {
            "url": url,
            "title": title or url,
            "snippet": snippet,
            "links": links,
        }

    def _newspaper_extract(self, url: str, settings: dict[str, Any], text_chars: int) -> dict[str, str] | None:
        if not bool(settings.get("newspaper_enabled", True)):
            return None
        try:
            from newspaper import Article, Config
        except Exception:
            try:
                # Compatibility path for forks/distributions that expose a different top-level module.
                from newspaper4k import Article, Config  # type: ignore
            except Exception:
                return None

        cfg = Config()
        cfg.fetch_images = False
        cfg.memoize_articles = False
        timeout_hint = int(settings.get("crawl_timeout_sec", 20) or 20)
        cfg.request_timeout = max(5, min(timeout_hint if timeout_hint > 0 else 20, 120))
        lang = str(settings.get("newspaper_language", "")).strip()
        if lang:
            cfg.language = lang

        article = Article(url=url, config=cfg)
        try:
            article.download()
            article.parse()
        except Exception:
            return None
        text = " ".join(str(article.text or "").split()).strip()
        if not text:
            return None
        if len(text) > text_chars:
            cut = text[:text_chars].rsplit(" ", 1)[0].strip()
            text = (cut or text[:text_chars]).strip() + "..."
        title = " ".join(str(article.title or "").split()).strip()
        return {"title": title, "snippet": text}

    def _fetch_page(self, url: str, settings: dict[str, Any], text_chars: int) -> dict[str, Any]:
        timeout_sec = int(settings.get("crawl_timeout_sec", 0))
        retry_attempts = int(settings.get("crawl_retry_attempts", 3))
        page: dict[str, Any]

        # Reddit-specific path: use JSON API to bypass JS rendering and login walls
        if self._is_reddit_url(url):
            try:
                reddit_page = self._fetch_reddit_json(url, text_chars=text_chars)
                if reddit_page and str(reddit_page.get("snippet", "")).strip():
                    return reddit_page
            except Exception:
                pass
            # Fall through to standard crawl as last resort

        crawl4ai_ready = bool(settings.get("crawl4ai_enabled", True)) and time.time() >= self._crawl4ai_backoff_until
        if crawl4ai_ready:
            try:
                page = self._fetch_page_crawl4ai(url=url, settings=settings, text_chars=text_chars)
            except Exception:
                self._crawl4ai_backoff_until = max(self._crawl4ai_backoff_until, time.time() + 120.0)
                page = self._fetch_page_basic(
                    url=url,
                    timeout_sec=timeout_sec,
                    text_chars=text_chars,
                    retry_attempts=retry_attempts,
                )
        else:
            page = self._fetch_page_basic(
                url=url,
                timeout_sec=timeout_sec,
                text_chars=text_chars,
                retry_attempts=retry_attempts,
            )

        parsed = self._newspaper_extract(url=url, settings=settings, text_chars=text_chars)
        if isinstance(parsed, dict):
            if parsed.get("title"):
                page["title"] = str(parsed.get("title", "")).strip()
            if parsed.get("snippet"):
                page["snippet"] = str(parsed.get("snippet", "")).strip()
        return page

    def _query_terms(self, query: str) -> set[str]:
        _STOPWORDS = {
            "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
            "have", "has", "had", "do", "does", "did", "will", "would", "shall",
            "should", "may", "might", "must", "can", "could", "of", "in", "on",
            "at", "to", "for", "with", "by", "from", "up", "about", "into",
            "what", "when", "where", "who", "which", "how", "that", "this",
            "and", "or", "not", "it", "its", "i", "me", "my", "you", "your",
        }
        tokens = set(re.split(r"[^a-z0-9]+", query.lower()))
        tokens.discard("")
        return tokens - _STOPWORDS

    def _link_relevance_score(self, url: str, query_terms: set[str]) -> float:
        """Score a candidate child URL for relevance to query terms (0.0–1.0).
        Returns 0.0 for navigation/structural URLs. Links below
        crawl_relevance_min_score are skipped when gating is enabled."""
        if not url:
            return 0.0
        parsed = urllib.parse.urlsplit(url)
        path = parsed.path.lower()
        query_str = parsed.query.lower()
        path_and_query = path + ("?" + query_str if query_str else "")
        _NAV_PATTERNS = (
            "/login", "/signin", "/signup", "/register", "/logout",
            "/about", "/contact", "/privacy", "/terms", "/cookie",
            "/search", "/tag/", "/tags/", "/category/", "/categories/",
            "/author/", "/feed", "/rss", "/sitemap",
            "?page=", "&page=", "?p=", "&p=",
            "/cdn-cgi/", "/wp-admin", "/wp-login",
        )
        for pat in _NAV_PATTERNS:
            if pat in path_and_query:
                return 0.0
        if not query_terms:
            return 0.5
        path_tokens = set(re.split(r"[^a-z0-9]+", path))
        path_tokens.discard("")
        matches = len(query_terms & path_tokens)
        score = min(1.0, (matches / len(query_terms)) * 0.8)
        if re.search(r"/\d{4}/\d{2}/", path):
            score = min(1.0, score + 0.2)
        elif len(path.split("/")) >= 4 and len(path) > 25:
            score = min(1.0, score + 0.1)
        return round(score, 3)

    def _crawl_sources(
        self,
        seeds: list[dict[str, str]],
        settings: dict[str, Any],
        query: str = "",
        *,
        exclude_urls: set[str] | None = None,
        on_source_crawled=None,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int]:
        depth_limit = int(settings.get("crawl_depth", 2))
        max_pages = int(settings.get("crawl_max_pages", 18))
        links_per_page = int(settings.get("crawl_links_per_page", 8))
        text_chars = int(settings.get("crawl_text_chars", 800))
        same_domain_only = bool(settings.get("crawl_same_domain_only", True))
        relevance_gating = bool(settings.get("crawl_relevance_gating_enabled", False))
        relevance_min_score = float(settings.get("crawl_relevance_min_score", 0.1))
        query_terms = self._query_terms(query) if relevance_gating else set()

        queue: deque[tuple[str, int, str]] = deque()
        enqueued: set[str] = set()
        visited: set[str] = {
            self._normalize_url(str(url).strip())
            for url in (exclude_urls or set())
            if self._normalize_url(str(url).strip())
        }
        pages: list[dict[str, Any]] = []
        failures: list[dict[str, Any]] = []
        gated_links: int = 0

        for row in seeds:
            seed_url = self._normalize_url(str(row.get("url", "")).strip())
            if not seed_url:
                continue
            host = self._hostname(seed_url)
            if not host:
                continue
            if seed_url in enqueued:
                continue
            queue.append((seed_url, 0, host))
            enqueued.add(seed_url)

        while queue and len(pages) < max_pages:
            current_url, depth, root_host = queue.popleft()
            if current_url in visited:
                continue
            visited.add(current_url)
            if not self._can_crawl_url(current_url):
                continue
            try:
                page = self._fetch_page(
                    url=current_url,
                    settings=settings,
                    text_chars=text_chars,
                )
                page["depth"] = depth
                page["root_host"] = root_host
                # Embedding-based paragraph filtering: keep only content relevant to the query
                if bool(settings.get("embedding_content_filter_enabled", True)) and query and page.get("snippet"):
                    try:
                        filtered = self._embedding_filter_content(query, str(page["snippet"]))
                        if filtered:
                            page["snippet"] = filtered
                    except Exception:
                        pass
                pages.append(page)
                # Fire per-source callback so the UI can show live discovery bubbles
                if callable(on_source_crawled):
                    try:
                        on_source_crawled({
                            "url": current_url,
                            "domain": self._hostname(current_url),
                            "title": str(page.get("title", "")).strip(),
                            "depth": depth,
                        })
                    except Exception:
                        pass
            except Exception as exc:
                failures.append({"url": current_url, "depth": depth, "error": str(exc)})
                continue

            if depth >= depth_limit:
                continue

            child_count = 0
            for href in page.get("links", []):
                if child_count >= links_per_page or len(enqueued) >= (max_pages * (links_per_page + 1)):
                    break
                next_url = self._normalize_url(str(href), base_url=current_url)
                if not next_url or next_url in enqueued or next_url in visited:
                    continue
                if not self._can_crawl_url(next_url):
                    continue
                if same_domain_only and self._hostname(next_url) != root_host:
                    continue
                if relevance_gating:
                    rel_score = self._link_relevance_score(next_url, query_terms)
                    if rel_score < relevance_min_score:
                        gated_links += 1
                        continue
                queue.append((next_url, depth + 1, root_host))
                enqueued.add(next_url)
                child_count += 1

        return pages, failures, gated_links

    def _append_source_log(self, payload: dict[str, Any]) -> None:
        line = json.dumps(payload, ensure_ascii=True)
        with self.sources_log_path.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")

    def _should_bypass_query_cache(self, query: str, note: str, explicit_bypass: bool) -> bool:
        if explicit_bypass:
            return True
        raw = f"{query} {note}".lower()
        return any(token in raw for token in ("refresh", "again"))

    def _query_cache_ttl_sec(self, query: str, topic_type: str) -> int:
        low = str(query or "").lower()
        live_markers = (
            "fight night",
            "live score",
            "live result",
            "live update",
            "who won",
            "score right now",
            "tonight",
            "main event",
            "kickoff",
            "tipoff",
        )
        if any(marker in low for marker in live_markers):
            return 10 * 60
        volatility = classify_fact_volatility(query, topic_type, query)
        if volatility in {"volatile", "semi_volatile"}:
            return 2 * 60 * 60
        return 24 * 60 * 60

    def _query_cache_settings(self, settings: dict[str, Any]) -> dict[str, Any]:
        # Keep the digest stable and tied to result-shaping controls.
        keys = (
            "provider",
            "max_results",
            "query_expansion_enabled",
            "query_expansion_variants",
            "query_decomposition_enabled",
            "query_decomposition_max_sub",
            "crawl_enabled",
            "crawl_depth",
            "crawl_max_pages",
            "crawl_links_per_page",
            "crawl_timeout_sec",
            "context_min_source_score",
            "source_scoring_enabled",
            "conflict_detection_enabled",
            "iterative_search_enabled",
            "iterative_search_time_budget_sec",
        )
        out: dict[str, Any] = {}
        for key in keys:
            if key in settings:
                out[key] = settings.get(key)
        return out

    @staticmethod
    def _cache_disclosure(age_sec: int) -> str:
        mins = max(0, int(round(age_sec / 60.0)))
        return f"Last retrieved {mins} minute{'s' if mins != 1 else ''} ago."

    def _run_query_with_cache(
        self,
        *,
        project: str,
        lane: str,
        query: str,
        reason: str,
        request_id: str,
        note: str,
        topic_type: str,
        settings: dict[str, Any],
        resolved_topic: str,
        bypass_cache: bool,
        progress_callback=None,
    ) -> dict[str, Any]:
        try:
            self._query_cache.purge_expired()
        except Exception:
            pass
        settings_key = build_settings_digest(self._query_cache_settings(settings))
        key = build_cache_key(
            project=project,
            query=normalize_cache_query(query),
            topic_type=resolved_topic,
            settings_hash=settings_key,
        )
        ttl_sec = self._query_cache_ttl_sec(query, resolved_topic)
        use_cache = not self._should_bypass_query_cache(query, note, bypass_cache)
        if use_cache:
            cached = self._query_cache.get(key)
            if isinstance(cached, dict):
                age_sec = int((cached.get("_cache", {}) or {}).get("age_sec", 0) or 0)
                result = dict(cached)
                result["cache_hit"] = True
                result["cache_key"] = key
                result["cache_ttl_sec"] = ttl_sec
                result["message"] = (
                    f"{str(result.get('message', '')).strip()} "
                    f"{self._cache_disclosure(age_sec)}"
                ).strip()
                result["cache_disclosure"] = self._cache_disclosure(age_sec)
                return result

        result = self._run_query_inner(
            project=project,
            lane=lane,
            query=query,
            reason=reason,
            request_id=request_id,
            note=note,
            topic_type=topic_type,
            settings=settings,
            resolved_topic=resolved_topic,
            progress_callback=progress_callback,
        )
        if bool(result.get("ok", False)):
            self._query_cache.put(
                key,
                result,
                ttl_sec=ttl_sec,
                topic_type=resolved_topic,
                source="run_quick_query" if self._is_quick_lookup_note(note) else "run_query",
            )
        result["cache_hit"] = False
        result["cache_key"] = key
        result["cache_ttl_sec"] = ttl_sec
        result["cache_disclosure"] = ""
        return result

    def run_query(
        self,
        *,
        project: str,
        lane: str,
        query: str,
        reason: str,
        request_id: str = "",
        note: str = "",
        topic_type: str = "general",
        bypass_cache: bool = False,
        progress_callback=None,
    ) -> dict[str, Any]:
        settings = self._load_settings()
        resolved_topic = self._resolve_topic_type(query, topic_type)
        # Activate TOR proxy for underground queries (if enabled in settings)
        self._tor_active = str(resolved_topic).strip().lower() == "underground"
        try:
            return self._run_query_with_cache(
                project=project, lane=lane, query=query, reason=reason,
                request_id=request_id, note=note, topic_type=topic_type,
                settings=settings, resolved_topic=resolved_topic,
                bypass_cache=bypass_cache,
                progress_callback=progress_callback,
            )
        finally:
            self._tor_active = False

    def run_quick_query(
        self,
        *,
        project: str,
        lane: str,
        query: str,
        reason: str,
        request_id: str = "",
        note: str = "",
        topic_type: str = "general",
        bypass_cache: bool = False,
        progress_callback=None,
    ) -> dict[str, Any]:
        settings = dict(self._load_settings())
        resolved_topic = self._resolve_topic_type(query, topic_type)
        # Quick chat lookups keep the same scoring and conflict logic, but trim
        # expansion/crawl breadth so the conversation layer can ground itself fast.
        settings["max_results"] = min(int(settings.get("max_results", 8) or 8), 4)
        settings["query_expansion_enabled"] = True
        settings["query_expansion_variants"] = 2
        settings["query_decomposition_enabled"] = True
        settings["query_decomposition_max_sub"] = 3  # cap at 3 sub-queries for chat speed
        settings["min_quality_sources"] = 1
        settings["context_min_source_score"] = min(float(settings.get("context_min_source_score", 0.62) or 0.62), 0.46)
        settings["crawl_enabled"] = True
        settings["crawl_depth"] = min(int(settings.get("crawl_depth", 2) or 2), 1)
        settings["crawl_max_pages"] = min(int(settings.get("crawl_max_pages", 18) or 18), 6)
        settings["crawl_links_per_page"] = min(int(settings.get("crawl_links_per_page", 8) or 8), 3)
        settings["crawl_timeout_sec"] = min(int(settings.get("crawl_timeout_sec", 12) or 12), 8)
        # Quick mode: allow iterative search only with a reduced time budget
        settings["iterative_search_time_budget_sec"] = 12
        self._tor_active = str(resolved_topic).strip().lower() == "underground"
        try:
            return self._run_query_with_cache(
                project=project,
                lane=lane,
                query=query,
                reason=reason,
                request_id=request_id,
                note=(str(note or "").strip() + " quick_chat_lookup").strip(),
                topic_type=topic_type,
                settings=settings,
                resolved_topic=resolved_topic,
                bypass_cache=bypass_cache,
                progress_callback=progress_callback,
            )
        finally:
            self._tor_active = False

    def _run_query_inner(
        self,
        *,
        project: str,
        lane: str,
        query: str,
        reason: str,
        request_id: str,
        note: str,
        topic_type: str,
        settings: dict[str, Any],
        resolved_topic: str,
        progress_callback=None,
    ) -> dict[str, Any]:
        _query_start_time = time.time()
        quick_lookup = self._is_quick_lookup_note(note)
        fresh_runs_enabled = bool(settings.get("fresh_runs_enabled", True)) and not quick_lookup
        try:
            fresh_history_limit = int(settings.get("fresh_runs_history_limit", 6))
        except (TypeError, ValueError):
            fresh_history_limit = 6
        try:
            fresh_min_new_domains = int(settings.get("fresh_runs_min_new_domains", 4))
        except (TypeError, ValueError):
            fresh_min_new_domains = 4
        recent_domains = (
            self._recent_source_domains_for_project(project, limit=fresh_history_limit)
            if fresh_runs_enabled
            else set()
        )
        fresh_run_stats: dict[str, Any] = {
            "recent_domains_considered": len(recent_domains),
            "novel_domains": 0,
            "novel_seed_rows": 0,
            "strict_novel_only": False,
        }
        pre_crawl_selection_stats: dict[str, Any] = {
            "enabled": bool(settings.get("pre_crawl_seed_selection_enabled", True)),
            "results_per_query": int(settings.get("pre_crawl_results_per_query", 20) or 20),
            "primary_quota": int(settings.get("pre_crawl_primary_quota", 5) or 5),
            "extra_quota_min": int(settings.get("pre_crawl_extra_quota_min", 2) or 2),
            "extra_quota_max": int(settings.get("pre_crawl_extra_quota_max", 3) or 3),
            "seed_count_before": 0,
            "seed_count_after": 0,
            "selected_by_variant": [],
            "primary_variant": "",
        }

        def _on_source(source_dict: dict[str, Any]) -> None:
            if callable(progress_callback):
                try:
                    progress_callback("web_source_discovered", source_dict)
                except Exception:
                    pass
        # Decompose compound queries (e.g. "Brand A vs Brand B vs Brand C prices") into
        # independent sub-queries before expansion so each entity gets its own searches.
        max_sub = int(settings.get("query_decomposition_max_sub", 5))
        sub_queries = self._decompose_query(query, settings, max_sub=max_sub)

        # For each sub-query generate expansion variants, then deduplicate the combined list
        expansion_variants = int(settings.get("query_expansion_variants", 4))
        all_variants: list[str] = []
        seen_variants: set[str] = set()
        for sq in sub_queries:
            for v in self._expand_queries(sq, settings, topic_type=resolved_topic):
                key = v.lower()
                if key not in seen_variants:
                    seen_variants.add(key)
                    all_variants.append(v)
        variant_queries = all_variants[:max(8, expansion_variants * len(sub_queries))]
        smart_query_variants: list[str] = []
        if not quick_lookup:
            smart_query_variants = self._smart_query_variants(
                project=project,
                query=query,
                settings=settings,
                existing_queries=variant_queries,
                recent_domains=recent_domains,
                topic_type=resolved_topic,
            )
            if smart_query_variants:
                variant_queries = self._merge_query_lists(variant_queries, smart_query_variants)
                variant_cap = max(8, (expansion_variants * len(sub_queries)) + len(smart_query_variants))
                variant_queries = variant_queries[:max(variant_cap, len(smart_query_variants))]
        if not variant_queries:
            return {
                "ok": False,
                "project": project,
                "lane": lane,
                "query": query,
                "topic_type": resolved_topic,
                "reason": reason,
                "request_id": request_id,
                "source_count": 0,
                "sources": [],
                "source_path": "",
                "query_expansion_enabled": bool(settings.get("query_expansion_enabled", True)),
                "query_variants_count": 0,
                "query_variants": [],
                "smart_query_variants": smart_query_variants,
                "fresh_runs_enabled": fresh_runs_enabled,
                "pre_crawl_selection": pre_crawl_selection_stats,
                "variant_hits": [],
                "source_scoring_enabled": bool(settings.get("source_scoring_enabled", True)),
                "source_scoring_summary": {
                    "enabled": bool(settings.get("source_scoring_enabled", True)),
                    "applied": False,
                    "strategy": "developer_domain_tiers_v1",
                    "tier_counts": {"tier1": 0, "tier2": 0, "tier3": 0, "banned_sources": 0},
                    "top_score": 0.0,
                },
                "conflict_detection_enabled": bool(settings.get("conflict_detection_enabled", True)),
                "conflict_summary": {"enabled": bool(settings.get("conflict_detection_enabled", True)), "applied": False, "conflict_count": 0, "conflicts": [], "note": ""},
                "crawl_relevance_gating_enabled": bool(settings.get("crawl_relevance_gating_enabled", False)),
                "crawl_gated_links": 0,
                "message": "Query is empty after normalization.",
            }

        max_results = max(1, min(int(settings.get("max_results", 8)), 20))
        if quick_lookup:
            search_results_per_variant = max_results
        else:
            try:
                pre_crawl_results_per_query = int(settings.get("pre_crawl_results_per_query", 20))
            except (TypeError, ValueError):
                pre_crawl_results_per_query = 20
            search_results_per_variant = max(20, pre_crawl_results_per_query)
        search_results_per_variant = max(1, min(search_results_per_variant, 20))
        seed_limit = max(
            search_results_per_variant,
            min(search_results_per_variant * max(1, len(variant_queries)), 240),
        )
        seeds: list[dict[str, str]] = []
        variant_hits: list[dict[str, Any]] = []
        for variant in variant_queries:
            rows = self.search(variant, max_results=search_results_per_variant)
            tagged_rows: list[dict[str, str]] = []
            for row in rows:
                payload = dict(row)
                payload["query_variant"] = variant
                tagged_rows.append(payload)
            seeds = self._merge_results(seeds, tagged_rows, seed_limit)
            variant_hits.append({"query": variant, "seed_hits": len(rows)})

        # --- Reddit direct search for product/review queries ---
        # Reddit returns honest community reviews but often doesn't surface in SearXNG/DDG
        if self._is_product_query(query) and not self._tor_active:
            try:
                reddit_seeds = self._reddit_search(query, limit=4)
                if reddit_seeds:
                    for row in reddit_seeds:
                        row["query_variant"] = f"reddit:{query}"
                    seeds = self._merge_results(seeds, reddit_seeds, seed_limit)
            except Exception:
                pass

        # --- Quality boost: if well-scoring T1+T2 seeds are scarce, search wider ---
        # Score seeds now so the boost condition reflects actual usability, not just domain membership.
        # A tier1/tier2 seed that scored below the quality floor is not useful and should not count.
        _boost_query_terms = self._query_terms(query)
        _boost_score_min = float(settings.get("context_min_source_score", 0.62))

        def _quality_tier12_count(seed_list: list[dict[str, str]]) -> int:
            count = 0
            for row in seed_list:
                tier = self._domain_tier(self._hostname(str(row.get("url", ""))))[0]
                if tier not in {"tier1", "tier2"}:
                    continue
                scored = self._score_one_source(row, _boost_query_terms, query=query, topic_type=resolved_topic)
                if not bool(scored.get("quality_blocked", False)) and float(scored.get("source_score", 0.0)) >= _boost_score_min:
                    count += 1
            return count

        _min_quality = max(1, int(settings.get("min_quality_sources", 2)))
        _boost_max_rounds = 2
        _boost_results = search_results_per_variant
        for _boost_round in range(_boost_max_rounds):
            if _quality_tier12_count(seeds) >= _min_quality:
                break
            _boost_seeds: list[dict[str, str]] = []
            for variant in variant_queries:
                for row in self.search(variant, max_results=_boost_results):
                    payload = dict(row)
                    payload["query_variant"] = variant
                    _boost_seeds.append(payload)
            _before = len(seeds)
            seeds = self._merge_results(seeds, _boost_seeds, seed_limit * 2)
            if len(seeds) == _before:
                break  # nothing new found, no point continuing
            _boost_results = min(20, _boost_results + max_results)  # widen slightly each round
        if fresh_runs_enabled and recent_domains and seeds:
            seeds, fresh_run_stats = self._prioritize_seed_domains_for_freshness(
                seeds,
                recent_domains,
                min_new_domains=fresh_min_new_domains,
            )
        if quick_lookup:
            pre_crawl_selection_stats["enabled"] = False
            pre_crawl_selection_stats["seed_count_before"] = len(seeds)
            pre_crawl_selection_stats["seed_count_after"] = len(seeds)
        else:
            seeds, pre_crawl_selection_stats = self._select_pre_crawl_seeds(
                seeds=seeds,
                query=query,
                variant_queries=variant_queries,
                settings=settings,
                resolved_topic=resolved_topic,
            )

        if not seeds:
            return {
                "ok": False,
                "project": project,
                "lane": lane,
                "query": query,
                "topic_type": resolved_topic,
                "reason": reason,
                "request_id": request_id,
                "source_count": 0,
                "sources": [],
                "source_path": "",
                "query_expansion_enabled": bool(settings.get("query_expansion_enabled", True)),
                "query_variants_count": len(variant_queries),
                "query_variants": variant_queries,
                "smart_query_variants": smart_query_variants,
                "fresh_runs_enabled": fresh_runs_enabled,
                "fresh_run_stats": fresh_run_stats,
                "pre_crawl_selection": pre_crawl_selection_stats,
                "variant_hits": variant_hits,
                "source_scoring_enabled": bool(settings.get("source_scoring_enabled", True)),
                "source_scoring_summary": {
                    "enabled": bool(settings.get("source_scoring_enabled", True)),
                    "applied": False,
                    "strategy": "developer_domain_tiers_v1",
                    "tier_counts": {"tier1": 0, "tier2": 0, "tier3": 0, "banned_sources": 0},
                    "top_score": 0.0,
                },
                "conflict_detection_enabled": bool(settings.get("conflict_detection_enabled", True)),
                "conflict_summary": {"enabled": bool(settings.get("conflict_detection_enabled", True)), "applied": False, "conflict_count": 0, "conflicts": [], "note": ""},
                "crawl_relevance_gating_enabled": bool(settings.get("crawl_relevance_gating_enabled", False)),
                "crawl_gated_links": 0,
                "message": "No web sources found (or network unavailable).",
            }

        provider = str(settings.get("provider", "auto")).strip().lower() or "auto"
        crawl_enabled = bool(settings.get("crawl_enabled", True))
        crawled_pages: list[dict[str, Any]] = []
        crawl_failures: list[dict[str, Any]] = []
        crawl_gated_links: int = 0
        second_pass_used = False
        second_pass_queries: list[str] = []
        second_pass_seed_hits: list[dict[str, Any]] = []
        second_pass_added_seeds = 0
        second_pass_added_pages = 0
        if crawl_enabled:
            crawled_pages, crawl_failures, crawl_gated_links = self._crawl_sources(
                seeds, settings, query=query, on_source_crawled=_on_source,
            )

        # Wikipedia guaranteed source — fetch via MediaWiki API for all topics except
        # underground/tor queries where anonymity is required.
        # Uses full plaintext article extract (not crawl4ai), so it's always clean.
        wiki_page: dict[str, Any] | None = None
        if resolved_topic != "underground":
            try:
                wiki_page = self._fetch_wikipedia_extract(query, text_chars=5000, topic_type=resolved_topic)
            except Exception:
                wiki_page = None
        if wiki_page:
            # Prepend so Wikipedia is always position-0; deduplication in scoring will
            # drop it later only if the same URL was already crawled from seeds.
            crawled_pages.insert(0, wiki_page)

        if crawled_pages:
            sources_raw: list[dict[str, Any]] = [
                {
                    "title": str(page.get("title", "")).strip(),
                    "url": str(page.get("url", "")).strip(),
                    "snippet": str(page.get("snippet", "")).strip(),
                    "depth": int(page.get("depth", 0)),
                }
                for page in crawled_pages
            ]
        else:
            sources_raw = [dict(row) for row in seeds]

        source_scoring_enabled = bool(settings.get("source_scoring_enabled", True))
        sources, source_scoring_summary = self._apply_source_scoring(
            sources=sources_raw,
            query=query,
            enabled=source_scoring_enabled,
            topic_type=resolved_topic,
        )
        quality_min_score = float(settings.get("context_min_source_score", 0.62))
        quality_min_score = max(0.1, min(quality_min_score, 1.0))
        raw_source_count = len(sources)
        quality_blocked_count = sum(1 for row in sources if bool(row.get("quality_blocked", False)))
        filtered_sources = [
            row
            for row in sources
            if not bool(row.get("quality_blocked", False))
            and float(row.get("source_score", 0.0)) >= quality_min_score
        ]
        if not filtered_sources:
            # If strong domains were captured but scored below threshold, keep a small fallback set.
            filtered_sources = [
                row
                for row in sources
                if not bool(row.get("quality_blocked", False))
                and str(row.get("source_tier", "tier3")) in {"tier1", "tier2"}
            ][:3]
        sources = filtered_sources
        source_scoring_summary["context_min_source_score"] = round(float(quality_min_score), 2)
        source_scoring_summary["quality_blocked_count"] = int(quality_blocked_count)
        source_scoring_summary["quality_filtered_out"] = max(0, raw_source_count - len(sources))
        post_filter_tiers = {"tier1": 0, "tier2": 0, "tier3": 0, "banned_sources": 0}
        for row in sources:
            tier = str(row.get("source_tier", "tier3"))
            if tier not in post_filter_tiers:
                tier = "tier3"
            post_filter_tiers[tier] += 1
        source_scoring_summary["post_filter_tier_counts"] = post_filter_tiers

        # --- LLM-based iterative search (layer 2) ---
        # Ask the LLM to identify gaps in pass-1 results and generate targeted follow-up queries.
        _time_budget = float(settings.get("iterative_search_time_budget_sec", 25.0))
        _iterative_enabled = bool(settings.get("iterative_search_enabled", True))
        _elapsed = time.time() - _query_start_time
        _llm_followup_used = False
        if _iterative_enabled and _elapsed < _time_budget and sources:
            try:
                followup_queries = self._generate_followup_queries(query, sources, settings)
                if followup_queries:
                    _existing_seed_urls = {
                        self._normalize_url(str(row.get("url", "")).strip())
                        for row in seeds
                        if self._normalize_url(str(row.get("url", "")).strip())
                    }
                    _existing_crawl_urls = {
                        self._normalize_url(str(row.get("url", "")).strip())
                        for row in crawled_pages
                        if self._normalize_url(str(row.get("url", "")).strip())
                    }
                    _followup_seeds: list[dict[str, str]] = []
                    for fq in followup_queries:
                        for row in self.search(fq, max_results=max(4, max_results)):
                            payload = dict(row)
                            payload["query_variant"] = fq
                            _followup_seeds.append(payload)
                        variant_hits.append({"query": fq, "seed_hits": len(_followup_seeds), "pass": "llm_followup"})
                    _new_seeds = [
                        row for row in _followup_seeds
                        if self._normalize_url(str(row.get("url", "")).strip()) not in _existing_seed_urls
                    ]
                    if _new_seeds:
                        seeds = self._merge_results(seeds, _followup_seeds, seed_limit * 2)
                        second_pass_queries = followup_queries
                        second_pass_used = True
                        second_pass_added_seeds = len(_new_seeds)
                        variant_queries = self._merge_query_lists(variant_queries, followup_queries)
                        if crawl_enabled:
                            _extra_pages, _extra_failures, _extra_gated = self._crawl_sources(
                                _new_seeds, settings, query=followup_queries[0],
                                exclude_urls=_existing_crawl_urls,
                                on_source_crawled=_on_source,
                            )
                            crawled_pages.extend(_extra_pages)
                            crawl_failures.extend(_extra_failures)
                            crawl_gated_links += _extra_gated
                            second_pass_added_pages = len(_extra_pages)
                        if crawled_pages:
                            _sources_raw2 = [
                                {
                                    "title": str(page.get("title", "")).strip(),
                                    "url": str(page.get("url", "")).strip(),
                                    "snippet": str(page.get("snippet", "")).strip(),
                                    "depth": int(page.get("depth", 0)),
                                }
                                for page in crawled_pages
                            ]
                            if wiki_page:
                                _wiki_norm = self._normalize_url(str(wiki_page.get("url", "")))
                                if not any(self._normalize_url(str(s.get("url", ""))) == _wiki_norm for s in _sources_raw2):
                                    _sources_raw2.insert(0, {
                                        "title": str(wiki_page.get("title", "")).strip(),
                                        "url": str(wiki_page.get("url", "")).strip(),
                                        "snippet": str(wiki_page.get("snippet", "")).strip(),
                                        "depth": 0,
                                    })
                        else:
                            _sources_raw2 = [dict(row) for row in seeds]
                        sources, source_scoring_summary = self._apply_source_scoring(
                            sources=_sources_raw2, query=query,
                            enabled=source_scoring_enabled, topic_type=resolved_topic,
                        )
                        _raw2 = len(sources)
                        _blocked2 = sum(1 for row in sources if bool(row.get("quality_blocked", False)))
                        _filtered2 = [
                            row for row in sources
                            if not bool(row.get("quality_blocked", False))
                            and float(row.get("source_score", 0.0)) >= quality_min_score
                        ]
                        if not _filtered2:
                            _filtered2 = [
                                row for row in sources
                                if not bool(row.get("quality_blocked", False))
                                and str(row.get("source_tier", "tier3")) in {"tier1", "tier2"}
                            ][:3]
                        sources = _filtered2
                        source_scoring_summary["context_min_source_score"] = round(float(quality_min_score), 2)
                        source_scoring_summary["quality_blocked_count"] = int(_blocked2)
                        source_scoring_summary["quality_filtered_out"] = max(0, _raw2 - len(sources))
                        _llm_followup_used = True
            except Exception:
                pass  # iterative pass is best-effort; never block the primary result

        # --- Heuristic refined second pass (fallback when LLM iterative pass didn't run) ---
        if not _llm_followup_used and self._should_run_refined_second_pass(
            query=query,
            resolved_topic=resolved_topic,
            seeds=seeds,
            sources=sources,
            crawled_pages=crawled_pages,
        ):
            second_pass_queries = self._refine_queries_for_second_pass(query, settings, topic_type=resolved_topic)
            if second_pass_queries:
                existing_seed_urls = {
                    self._normalize_url(str(row.get("url", "")).strip())
                    for row in seeds
                    if self._normalize_url(str(row.get("url", "")).strip())
                }
                existing_crawl_urls = {
                    self._normalize_url(str(row.get("url", "")).strip())
                    for row in crawled_pages
                    if self._normalize_url(str(row.get("url", "")).strip())
                }
                refined_seed_rows: list[dict[str, str]] = []
                refined_max_results = min(20, max(max_results + 2, 8))
                for variant in second_pass_queries:
                    rows = self.search(variant, max_results=refined_max_results)
                    second_pass_seed_hits.append({"query": variant, "seed_hits": len(rows), "pass": "refined"})
                    for row in rows:
                        payload = dict(row)
                        payload["query_variant"] = variant
                        refined_seed_rows.append(payload)
                merged_seeds = self._merge_results(seeds, refined_seed_rows, seed_limit * 2)
                new_seed_rows = [
                    row for row in merged_seeds
                    if self._normalize_url(str(row.get("url", "")).strip()) not in existing_seed_urls
                ]
                if new_seed_rows:
                    seeds = merged_seeds
                    second_pass_added_seeds = len(new_seed_rows)
                    variant_hits.extend(second_pass_seed_hits)
                    variant_queries = self._merge_query_lists(variant_queries, second_pass_queries)
                    second_pass_used = True
                    if crawl_enabled:
                        extra_pages, extra_failures, extra_gated_links = self._crawl_sources(
                            new_seed_rows,
                            settings,
                            query=second_pass_queries[0] if second_pass_queries else query,
                            exclude_urls=existing_crawl_urls,
                            on_source_crawled=_on_source,
                        )
                        crawled_pages.extend(extra_pages)
                        crawl_failures.extend(extra_failures)
                        crawl_gated_links += extra_gated_links
                        second_pass_added_pages = len(extra_pages)
                    if crawled_pages:
                        sources_raw = [
                            {
                                "title": str(page.get("title", "")).strip(),
                                "url": str(page.get("url", "")).strip(),
                                "snippet": str(page.get("snippet", "")).strip(),
                                "depth": int(page.get("depth", 0)),
                            }
                            for page in crawled_pages
                        ]
                        # Re-inject Wikipedia at position 0 if it was fetched but not crawled again
                        if wiki_page:
                            wiki_url_norm = self._normalize_url(str(wiki_page.get("url", "")))
                            if not any(
                                self._normalize_url(str(s.get("url", ""))) == wiki_url_norm
                                for s in sources_raw
                            ):
                                sources_raw.insert(0, {
                                    "title": str(wiki_page.get("title", "")).strip(),
                                    "url": str(wiki_page.get("url", "")).strip(),
                                    "snippet": str(wiki_page.get("snippet", "")).strip(),
                                    "depth": 0,
                                })
                    else:
                        sources_raw = [dict(row) for row in seeds]
                    sources, source_scoring_summary = self._apply_source_scoring(
                        sources=sources_raw,
                        query=query,
                        enabled=source_scoring_enabled,
                        topic_type=resolved_topic,
                    )
                    raw_source_count = len(sources)
                    quality_blocked_count = sum(1 for row in sources if bool(row.get("quality_blocked", False)))
                    filtered_sources = [
                        row
                        for row in sources
                        if not bool(row.get("quality_blocked", False))
                        and float(row.get("source_score", 0.0)) >= quality_min_score
                    ]
                    if not filtered_sources:
                        filtered_sources = [
                            row
                            for row in sources
                            if not bool(row.get("quality_blocked", False))
                            and str(row.get("source_tier", "tier3")) in {"tier1", "tier2"}
                        ][:3]
                    sources = filtered_sources
                    source_scoring_summary["context_min_source_score"] = round(float(quality_min_score), 2)
                    source_scoring_summary["quality_blocked_count"] = int(quality_blocked_count)
                    source_scoring_summary["quality_filtered_out"] = max(0, raw_source_count - len(sources))
                    post_filter_tiers = {"tier1": 0, "tier2": 0, "tier3": 0, "banned_sources": 0}
                    for row in sources:
                        tier = str(row.get("source_tier", "tier3"))
                        if tier not in post_filter_tiers:
                            tier = "tier3"
                        post_filter_tiers[tier] += 1
                    source_scoring_summary["post_filter_tier_counts"] = post_filter_tiers

        conflict_detection_enabled = bool(settings.get("conflict_detection_enabled", True))
        if not sources:
            return {
                "ok": False,
                "project": project,
                "lane": lane,
                "query": query,
                "topic_type": resolved_topic,
                "reason": reason,
                "request_id": request_id,
                "source_count": 0,
                "sources": [],
                "source_path": "",
                "provider": provider,
                "seed_count": len(seeds),
                "query_expansion_enabled": bool(settings.get("query_expansion_enabled", True)),
                "query_variants_count": len(variant_queries),
                "query_variants": variant_queries,
                "smart_query_variants": smart_query_variants,
                "variant_hits": variant_hits,
                "refined_second_pass_used": second_pass_used,
                "refined_second_pass_queries": second_pass_queries,
                "refined_second_pass_added_seeds": second_pass_added_seeds,
                "refined_second_pass_added_pages": second_pass_added_pages,
                "fresh_runs_enabled": fresh_runs_enabled,
                "fresh_run_stats": fresh_run_stats,
                "pre_crawl_selection": pre_crawl_selection_stats,
                "source_scoring_enabled": source_scoring_enabled,
                "source_scoring_summary": source_scoring_summary,
                "conflict_detection_enabled": conflict_detection_enabled,
                "conflict_summary": {
                    "enabled": conflict_detection_enabled,
                    "applied": False,
                    "conflict_count": 0,
                    "conflicts": [],
                    "note": "",
                    "topic_type": resolved_topic,
                },
                "crawl_relevance_gating_enabled": bool(settings.get("crawl_relevance_gating_enabled", False)),
                "crawl_gated_links": crawl_gated_links,
                "crawl_pages": len(crawled_pages),
                "crawl_failures": len(crawl_failures),
                "crawl_enabled": crawl_enabled,
                "message": (
                    "No high-confidence web sources passed relevance and quality filters. "
                    "Try a more specific query or add known trusted domains."
                ),
            }
        conflict_summary = self._detect_source_conflicts(
            sources=sources,
            query=query,
            enabled=conflict_detection_enabled,
            topic_type=resolved_topic,
        )

        # === Real Intelligence Behavior passes ===
        _intel_snippets = [str(s.get("snippet", "")) for s in sources if str(s.get("snippet", "")).strip()]
        _intel_ordered = [(str(s.get("source_domain", "")), str(s.get("snippet", ""))) for s in sources]
        _independence = SourceIndependenceScorer.score(_intel_snippets)
        _mutation = NarrativeMutationTracker.analyze(_intel_ordered)
        _consensus = ConsensusAlarmSystem.evaluate(sources)
        _semantic_contradictions = CrossDomainContradictionDetector.detect(sources)
        intel_summary: dict[str, Any] = {
            "independence": _independence,
            "mutation": _mutation,
            "consensus": _consensus,
            "semantic_contradictions": _semantic_contradictions,
        }

        # Compact alert strings — inserted at the top of the document so they
        # survive context-window trimming and are never "lost in the middle".
        _alert_lines: list[str] = []
        if _independence.get("warning"):
            _alert_lines.append(f"WIRE LAUNDERING SIGNAL: {_independence['warning']}")
        if _mutation.get("mutation_detected"):
            _alert_lines.append(
                f"NARRATIVE MUTATION (confidence={_mutation.get('confidence', 0.0):.2f}): "
                f"{_mutation.get('note', '')}"
            )
        if _consensus.get("alarm"):
            _alert_lines.append(f"CONSENSUS ALARM: {_consensus.get('reason', '')}")
        for _sc in _semantic_contradictions:
            _alert_lines.append(f"CONTRADICTION: {_sc.get('note', '')}")

        lines = [
            "# Web Research Source Cache",
            "",
            f"- request_id: {request_id or 'direct'}",
            f"- project: {project}",
            f"- lane: {lane}",
            f"- query: {query}",
            f"- reason: {reason}",
            f"- note: {note.strip() or 'none'}",
            f"- captured_at: {_now_iso()}",
            f"- seed_count: {len(seeds)}",
            f"- query_expansion_enabled: {bool(settings.get('query_expansion_enabled', True))}",
            f"- query_variants_count: {len(variant_queries)}",
            f"- query_variants: {' | '.join(variant_queries)}",
            f"- smart_query_variants: {' | '.join(smart_query_variants) if smart_query_variants else 'none'}",
            f"- fresh_runs_enabled: {fresh_runs_enabled}",
            f"- fresh_recent_domains_considered: {int(fresh_run_stats.get('recent_domains_considered', 0))}",
            f"- fresh_novel_domains: {int(fresh_run_stats.get('novel_domains', 0))}",
            f"- fresh_novel_seed_rows: {int(fresh_run_stats.get('novel_seed_rows', 0))}",
            f"- fresh_strict_novel_only: {bool(fresh_run_stats.get('strict_novel_only', False))}",
            f"- pre_crawl_seed_selection_enabled: {bool(pre_crawl_selection_stats.get('enabled', True))}",
            f"- pre_crawl_results_per_query: {int(pre_crawl_selection_stats.get('results_per_query', 20))}",
            f"- pre_crawl_primary_quota: {int(pre_crawl_selection_stats.get('primary_quota', 5))}",
            f"- pre_crawl_extra_quota_min: {int(pre_crawl_selection_stats.get('extra_quota_min', 2))}",
            f"- pre_crawl_extra_quota_max: {int(pre_crawl_selection_stats.get('extra_quota_max', 3))}",
            f"- pre_crawl_seed_count_before: {int(pre_crawl_selection_stats.get('seed_count_before', len(seeds)))}",
            f"- pre_crawl_seed_count_after: {int(pre_crawl_selection_stats.get('seed_count_after', len(seeds)))}",
            f"- refined_second_pass_used: {second_pass_used}",
            f"- refined_second_pass_queries: {' | '.join(second_pass_queries) if second_pass_queries else 'none'}",
            f"- refined_second_pass_added_seeds: {second_pass_added_seeds}",
            f"- refined_second_pass_added_pages: {second_pass_added_pages}",
            f"- provider: {provider}",
            f"- source_scoring_enabled: {source_scoring_enabled}",
            f"- source_scoring_applied: {bool(source_scoring_summary.get('applied', False))}",
            (
                "- source_tier_counts_t1_t2_t3_banned: "
                f"{source_scoring_summary.get('tier_counts', {}).get('tier1', 0)}/"
                f"{source_scoring_summary.get('tier_counts', {}).get('tier2', 0)}/"
                f"{source_scoring_summary.get('tier_counts', {}).get('tier3', 0)}/"
                f"{source_scoring_summary.get('tier_counts', {}).get('banned_sources', 0)}"
            ),
            f"- source_score_top: {float(source_scoring_summary.get('top_score', 0.0)):.2f}",
            f"- context_min_source_score: {float(source_scoring_summary.get('context_min_source_score', settings.get('context_min_source_score', 0.62))):.2f}",
            f"- quality_blocked_count: {int(source_scoring_summary.get('quality_blocked_count', 0))}",
            f"- quality_filtered_out: {int(source_scoring_summary.get('quality_filtered_out', 0))}",
            f"- conflict_detection_enabled: {conflict_detection_enabled}",
            f"- conflict_count: {int(conflict_summary.get('conflict_count', 0))}",
            f"- crawl_relevance_gating_enabled: {bool(settings.get('crawl_relevance_gating_enabled', False))}",
            f"- crawl_relevance_min_score: {float(settings.get('crawl_relevance_min_score', 0.1)):.2f}",
            f"- crawl_gated_links: {crawl_gated_links}",
            f"- crawl_enabled: {crawl_enabled}",
            f"- crawl4ai_enabled: {settings.get('crawl4ai_enabled', True)}",
            f"- newspaper_enabled: {settings.get('newspaper_enabled', True)}",
            f"- crawl_depth: {settings.get('crawl_depth', 2)}",
            f"- crawl_max_pages: {settings.get('crawl_max_pages', 18)}",
            f"- crawl_links_per_page: {settings.get('crawl_links_per_page', 8)}",
            f"- crawl_timeout_sec: {settings.get('crawl_timeout_sec', 12)}",
            f"- crawl_same_domain_only: {settings.get('crawl_same_domain_only', True)}",
            f"- crawl_pages_collected: {len(crawled_pages)}",
            f"- crawl_failures: {len(crawl_failures)}",
        ]
        if _alert_lines:
            lines[2:2] = ["## Active Warnings"] + _alert_lines + [""]
        lines.extend(["", "## Query Variant Hits"])
        for idx, row in enumerate(variant_hits, start=1):
            lines.append(f"{idx}. {row.get('query', '')} | seed_hits={int(row.get('seed_hits', 0))}")

        lines.extend(["", "## Pre-Crawl Selection"])
        selected_by_variant = (
            pre_crawl_selection_stats.get("selected_by_variant", [])
            if isinstance(pre_crawl_selection_stats.get("selected_by_variant", []), list)
            else []
        )
        if selected_by_variant:
            for idx, row in enumerate(selected_by_variant, start=1):
                variant = str(row.get("variant", "")).strip()
                selected = int(row.get("selected", 0) or 0)
                available = int(row.get("available", 0) or 0)
                quota = row.get("quota", None)
                quota_min = row.get("quota_min", None)
                quota_max = row.get("quota_max", None)
                if isinstance(quota, int):
                    lines.append(f"{idx}. {variant} | selected={selected}/{quota} | available={available}")
                else:
                    lines.append(
                        f"{idx}. {variant} | selected={selected} | quota_min={int(quota_min or 0)}"
                        f" quota_max={int(quota_max or 0)} | available={available}"
                    )
        else:
            lines.append("1. (selection disabled or no per-variant stats)")

        lines.extend(["", "## Seed Results"])
        for idx, row in enumerate(seeds, start=1):
            title = str(row.get("title", "")).strip() or str(row.get("url", "")).strip()
            url_value = str(row.get("url", "")).strip()
            snippet = str(row.get("snippet", "")).strip()
            source_variant = str(row.get("query_variant", "")).strip()
            if source_variant:
                lines.append(f"{idx}. [{title}]({url_value}) | variant={source_variant}")
            else:
                lines.append(f"{idx}. [{title}]({url_value})")
            if snippet:
                lines.append(f"   - {snippet}")

        lines.extend(
            [
                "",
                "## Traversed Pages" if crawled_pages else "## Traversed Pages (none)",
            ]
        )
        for idx, row in enumerate(crawled_pages, start=1):
            title = str(row.get("title", "")).strip() or str(row.get("url", "")).strip()
            url_value = str(row.get("url", "")).strip()
            snippet = str(row.get("snippet", "")).strip()
            depth = int(row.get("depth", 0))
            lines.append(f"{idx}. d={depth} [{title}]({url_value})")
            if snippet:
                lines.append(f"   - {snippet}")

        if crawl_failures:
            lines.extend(["", "## Crawl Failures (sample)"])
            for idx, row in enumerate(crawl_failures[:20], start=1):
                lines.append(f"{idx}. d={row.get('depth', 0)} {row.get('url', '')}")
                lines.append(f"   - error: {row.get('error', '')}")

        if int(conflict_summary.get("conflict_count", 0)) > 0:
            lines.extend(["", "## Preflight Conflict Flags"])
            for idx, row in enumerate(conflict_summary.get("conflicts", [])[:6], start=1):
                kind = str(row.get("type", "unknown"))
                lines.append(f"{idx}. type={kind} | source_coverage={int(row.get('source_coverage', 0))}")
                values = row.get("values", [])
                if isinstance(values, list):
                    for claim in values[:4]:
                        value = str(claim.get("value", "")).strip()
                        srcs = claim.get("sources", [])
                        lines.append(f"   - {value} | sources={srcs}")

        lines.extend(["", "## Intelligence Analysis"])
        lines.append(f"- source_independence: {float(_independence.get('independence_score', 1.0)):.2f}")
        if _independence.get("warning"):
            lines.append(f"  WARNING: {_independence['warning']}")
        if _mutation.get("mutation_detected"):
            lines.append(
                f"- narrative_mutation: DETECTED (confidence={float(_mutation.get('confidence', 0.0)):.2f})"
            )
            if _mutation.get("note"):
                lines.append(f"  note: {_mutation['note']}")
        else:
            lines.append("- narrative_mutation: none detected")
        if _consensus.get("alarm"):
            lines.append(
                f"- consensus_alarm: TOO-PERFECT"
                f" (uniformity={float(_consensus.get('uniformity_ratio', 0.0)):.2f},"
                f" tier1_anchor={_consensus.get('has_tier1_anchor', False)})"
            )
            lines.append(f"  WARNING: {_consensus.get('reason', '')}")
        else:
            lines.append(
                f"- consensus_alarm: none"
                f" (uniformity={float(_consensus.get('uniformity_ratio', 0.0)):.2f},"
                f" tier1_anchor={_consensus.get('has_tier1_anchor', False)})"
            )
        if _semantic_contradictions:
            lines.append(f"- semantic_contradictions: {len(_semantic_contradictions)}")
            for sc in _semantic_contradictions:
                lines.append(
                    f"  - subject='{sc.get('subject', '')}'"
                    f" | pos={sc.get('positive_sources', [])}"
                    f" vs neg={sc.get('negative_sources', [])}"
                )
        else:
            lines.append("- semantic_contradictions: none detected")

        lines.extend(
            [
                "",
                "## Sources Used By Orchestrator",
            ]
        )
        for idx, row in enumerate(sources, start=1):
            title = str(row.get("title", "")).strip() or str(row.get("url", "")).strip()
            url_value = str(row.get("url", "")).strip()
            snippet = str(row.get("snippet", "")).strip()
            depth = row.get("depth", None)
            score = float(row.get("source_score", 0.0))
            tier = str(row.get("source_tier", "tier3"))
            if isinstance(depth, int):
                lines.append(f"{idx}. d={depth} [{title}]({url_value}) | {tier} score={score:.2f}")
            else:
                lines.append(f"{idx}. [{title}]({url_value}) | {tier} score={score:.2f}")
            if snippet:
                lines.append(f"   - {snippet}")

        filename = self.store.timestamped_name("web_sources")
        source_path = self.store.write_project_file(project, "research_web_sources", filename, "\n".join(lines) + "\n")
        log_payload = {
            "ts": _now_iso(),
            "request_id": request_id,
            "project": project,
            "lane": lane,
            "query": query,
            "topic_type": resolved_topic,
            "reason": reason,
            "note": note.strip(),
            "source_path": str(source_path),
            "provider": provider,
            "seed_count": len(seeds),
            "query_expansion_enabled": bool(settings.get("query_expansion_enabled", True)),
            "query_variants_count": len(variant_queries),
            "query_variants": variant_queries,
            "smart_query_variants": smart_query_variants,
            "fresh_runs_enabled": fresh_runs_enabled,
            "fresh_run_stats": fresh_run_stats,
            "pre_crawl_selection": pre_crawl_selection_stats,
            "variant_hits": variant_hits,
            "refined_second_pass_used": second_pass_used,
            "refined_second_pass_queries": second_pass_queries,
            "refined_second_pass_added_seeds": second_pass_added_seeds,
            "refined_second_pass_added_pages": second_pass_added_pages,
            "source_scoring_enabled": source_scoring_enabled,
            "source_scoring_summary": source_scoring_summary,
            "conflict_detection_enabled": conflict_detection_enabled,
            "conflict_summary": conflict_summary,
            "crawl_relevance_gating_enabled": bool(settings.get("crawl_relevance_gating_enabled", False)),
            "crawl_gated_links": crawl_gated_links,
            "crawl_enabled": crawl_enabled,
            "crawl_pages": len(crawled_pages),
            "crawl_failures": len(crawl_failures),
            "intel_summary": intel_summary,
            "sources": sources,
        }
        self._append_source_log(log_payload)

        # Persist cleaned chunks to DB cache and prune stale data.
        # Both ops are best-effort — failures never block the query result.
        cache_ttl = int(settings.get("cache_ttl_days", 14))
        log_retain = int(settings.get("log_retain_days", 30))
        try:
            self._store_web_chunks(project, sources, ttl_days=cache_ttl)
        except Exception:
            pass
        try:
            self._purge_expired_web_chunks()
            self._purge_old_source_log(retain_days=log_retain)
        except Exception:
            pass

        return {
            "ok": True,
            "project": project,
            "lane": lane,
            "query": query,
            "topic_type": resolved_topic,
            "reason": reason,
            "request_id": request_id,
            "source_count": len(sources),
            "provider": provider,
            "seed_count": len(seeds),
            "query_expansion_enabled": bool(settings.get("query_expansion_enabled", True)),
            "query_variants_count": len(variant_queries),
            "query_variants": variant_queries,
            "smart_query_variants": smart_query_variants,
            "fresh_runs_enabled": fresh_runs_enabled,
            "fresh_run_stats": fresh_run_stats,
            "pre_crawl_selection": pre_crawl_selection_stats,
            "variant_hits": variant_hits,
            "refined_second_pass_used": second_pass_used,
            "refined_second_pass_queries": second_pass_queries,
            "refined_second_pass_added_seeds": second_pass_added_seeds,
            "refined_second_pass_added_pages": second_pass_added_pages,
            "source_scoring_enabled": source_scoring_enabled,
            "source_scoring_summary": source_scoring_summary,
            "conflict_detection_enabled": conflict_detection_enabled,
            "conflict_summary": conflict_summary,
            "crawl_relevance_gating_enabled": bool(settings.get("crawl_relevance_gating_enabled", False)),
            "crawl_gated_links": crawl_gated_links,
            "crawl_pages": len(crawled_pages),
            "crawl_failures": len(crawl_failures),
            "crawl_enabled": crawl_enabled,
            "intel_summary": intel_summary,
            "sources": sources,
            "source_path": str(source_path),
            "message": (
                f"Provider '{provider}' captured {len(seeds)} seed source(s), traversed {len(crawled_pages)} page(s), "
                f"usable source context entries: {len(sources)}."
            ),
        }

    def approve_and_run(self, request_id: str, note: str = "") -> dict[str, Any] | None:
        key = request_id.strip()
        with self.lock:
            rows = self._load_pending()
            target: dict[str, Any] | None = None
            for row in rows:
                if str(row.get("id", "")) != key:
                    continue
                if str(row.get("status", "")).lower() != "open":
                    return None
                target = row
                break
            if target is None:
                return None

        result = self.run_query(
            project=str(target.get("project", "general")),
            lane=str(target.get("lane", "project")),
            query=str(target.get("query", "")),
            reason=str(target.get("reason", "")),
            request_id=key,
            note=note,
            topic_type=str(target.get("topic_type", "general")),
        )

        with self.lock:
            rows = self._load_pending()
            hit: dict[str, Any] | None = None
            for row in rows:
                if str(row.get("id", "")) != key:
                    continue
                if str(row.get("status", "")).lower() != "open":
                    return None
                row["status"] = "resolved"
                row["answer_note"] = note.strip()
                row["updated_at"] = _now_iso()
                row["resolved_at"] = _now_iso()
                row["source_count"] = int(result.get("source_count", 0))
                row["source_path"] = str(result.get("source_path", ""))
                row["run_ok"] = bool(result.get("ok", False))
                hit = row
                break
            if hit is None:
                return None
            self._save_pending(rows)

        result["pending"] = hit
        return result

    def recent_sources_for_project(self, project: str, limit: int = 8) -> list[dict[str, Any]]:
        key = project.strip()
        limit = max(1, min(limit, 100))
        rows: list[dict[str, Any]] = []
        if not self.sources_log_path.exists():
            return rows
        for line in self.sources_log_path.read_text(encoding="utf-8", errors="ignore").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue
            if str(data.get("project", "")) != key:
                continue
            if not isinstance(data, dict):
                continue
            rows.append(data)
        rows.sort(key=lambda x: str(x.get("ts", "")), reverse=True)
        return rows[:limit]

    def web_context_for_project(self, project: str, limit: int = 6) -> str:
        logs = self.recent_sources_for_project(project, limit=limit)
        settings = self._load_settings()
        min_score = max(0.1, min(float(settings.get("context_min_source_score", 0.62)), 1.0))
        lines = ["Recent web source cache (use only if relevant):"]

        # Surface any source-integrity warnings from the most recent run so research
        # agents see them before processing source snippets.
        if logs:
            _latest_intel = logs[0].get("intel_summary", {}) if isinstance(logs[0].get("intel_summary"), dict) else {}
            _integrity_warnings: list[str] = []
            _ind_warn = str(_latest_intel.get("independence", {}).get("warning", "") or "").strip()
            if _ind_warn:
                _integrity_warnings.append(f"SOURCE INTEGRITY — Wire laundering signal: {_ind_warn}")
            _mut = _latest_intel.get("mutation", {}) if isinstance(_latest_intel.get("mutation"), dict) else {}
            if _mut.get("mutation_detected"):
                _mut_note = str(_mut.get("note", "") or "").strip()
                _mut_conf = float(_mut.get("confidence", 0.0) or 0.0)
                _integrity_warnings.append(
                    f"SOURCE INTEGRITY — Narrative mutation detected (confidence={_mut_conf:.2f}): {_mut_note}"
                )
            if _integrity_warnings:
                lines.append("## Source Integrity Warnings")
                lines.extend(_integrity_warnings)
                lines.append("## Sources")

        count = 0

        for log in logs:
            for source in log.get("sources", []) if isinstance(log.get("sources"), list) else []:
                if count >= limit:
                    break
                title = str(source.get("title", "")).strip()
                url_value = str(source.get("url", "")).strip()
                if not url_value:
                    continue
                snippet = str(source.get("snippet", "")).strip()
                _, _, inferred_blocked = self._low_signal_penalty(
                    url=url_value,
                    title=title,
                    snippet=snippet,
                    query_terms=set(),
                )
                quality_blocked = bool(source.get("quality_blocked", False)) or bool(inferred_blocked)
                if quality_blocked:
                    continue
                score = float(source.get("source_score", 0.0))
                if score < min_score:
                    continue
                depth = source.get("depth", None)
                tier = str(source.get("source_tier", "tier3")).strip() or "tier3"
                freshness = float(source.get("freshness_score", 0.0))
                if isinstance(depth, int):
                    lines.append(
                        f"- d={depth} [{tier} {score:.2f} fresh={freshness:.2f}] {title or url_value} | {url_value}"
                    )
                else:
                    lines.append(f"- [{tier} {score:.2f} fresh={freshness:.2f}] {title or url_value} | {url_value}")
                if snippet:
                    lines.append(f"  snippet: {snippet}")
                count += 1
            if count >= limit:
                break

        # Fall back to persistent DB cache when the live log is empty or sparse.
        if count < limit:
            seen_urls: set[str] = {
                str(s.get("url", ""))
                for log in logs
                for s in (log.get("sources", []) if isinstance(log.get("sources"), list) else [])
            }
            try:
                cached = self._query_cached_chunks(project, limit=limit - count)
                for row in cached:
                    url_value = str(row.get("url", "")).strip()
                    if not url_value or url_value in seen_urls:
                        continue
                    title = str(row.get("title", "")).strip()
                    snippet = str(row.get("snippet", "")).strip()
                    tier = str(row.get("source_tier", "tier3")).strip() or "tier3"
                    score = float(row.get("source_score", 0.0))
                    crawled_at = str(row.get("crawled_at", "")).split("T")[0]
                    lines.append(f"- [cached {tier} {score:.2f} as-of={crawled_at}] {title or url_value} | {url_value}")
                    if snippet:
                        lines.append(f"  snippet: {snippet}")
                    count += 1
                    seen_urls.add(url_value)
            except Exception:
                pass

        if count == 0:
            return ""
        return "\n".join(lines)

    def sources_text(self, project: str, limit: int = 10) -> str:
        logs = self.recent_sources_for_project(project, limit=limit)
        if not logs:
            return f"No web source cache yet for project '{project}'."
        lines = [f"Recent web source cache for '{project}' ({len(logs)} runs):"]
        for row in logs:
            ts = str(row.get("ts", ""))
            query = str(row.get("query", ""))
            source_path = str(row.get("source_path", ""))
            sources = row.get("sources", []) if isinstance(row.get("sources"), list) else []
            seed_count = int(row.get("seed_count", 0))
            crawl_pages = int(row.get("crawl_pages", 0))
            crawl_failures = int(row.get("crawl_failures", 0))
            conflict_summary = row.get("conflict_summary", {}) if isinstance(row.get("conflict_summary", {}), dict) else {}
            conflict_count = int(conflict_summary.get("conflict_count", 0))
            lines.append(
                f"- [{ts}] query={query} | used={len(sources)} | seeds={seed_count} | "
                f"crawl_pages={crawl_pages} | crawl_failures={crawl_failures} | "
                f"conflicts={conflict_count} | file={source_path}"
            )
        return "\n".join(lines)




# ============================================================
# REAL INTELLIGENCE BEHAVIORS
# Five active analysis passes run on every query result set.
# ============================================================



class NarrativeMutationTracker:
    """
    Detects how claims mutate as they propagate across sources.

    Early sources tend to hedge ("may", "reportedly", "alleged").
    Later sources that copy-paste or paraphrase often drop hedges and
    inflate certainty ("confirmed", "revealed", "is definitively").

    This is the core fingerprint of citation laundering and PR cascade:
    a claim that starts uncertain and becomes "fact" with no new evidence.
    """

    HEDGE_TERMS = {
        "may", "might", "could", "reportedly", "allegedly", "sources say",
        "claims", "appears to", "seems", "possible", "suggests", "unconfirmed",
        "according to some", "rumored", "speculated", "believed to",
    }
    CERTAINTY_TERMS = {
        "confirmed", "proved", "proven", "revealed", "officially",
        "definitively", "announced", "stated", "declared", "established",
        "verified", "fact", "undeniably",
    }

    @classmethod
    def analyze(cls, ordered_snippets: list[tuple[str, str]]) -> dict[str, Any]:
        """
        ordered_snippets: [(domain, text), ...] in discovery order (earliest first).
        Returns mutation report.
        """
        if len(ordered_snippets) < 2:
            return {"mutation_detected": False, "confidence": 0.0, "note": ""}

        hedge_counts: list[int] = []
        certainty_counts: list[int] = []
        for _domain, text in ordered_snippets:
            tl = text.lower()
            hedge_counts.append(sum(1 for t in cls.HEDGE_TERMS if t in tl))
            certainty_counts.append(sum(1 for t in cls.CERTAINTY_TERMS if t in tl))

        mid = max(1, len(ordered_snippets) // 2)
        early_hedges = sum(hedge_counts[:mid])
        late_hedges = sum(hedge_counts[mid:])
        early_cert = sum(certainty_counts[:mid])
        late_cert = sum(certainty_counts[mid:])

        mutation = (early_hedges > late_hedges) and (late_cert > early_cert)
        if mutation:
            confidence = round(min(1.0, (early_hedges - late_hedges) * 0.15 + (late_cert - early_cert) * 0.1), 3)
            note = (
                f"Hedging language dropped from {early_hedges} to {late_hedges} hits; "
                f"certainty language rose from {early_cert} to {late_cert} hits. "
                "Claim may have been laundered into fact without new primary evidence."
            )
        else:
            confidence = 0.0
            note = ""

        return {
            "mutation_detected": mutation,
            "confidence": confidence,
            "early_hedge_count": early_hedges,
            "late_certainty_count": late_cert,
            "note": note,
        }


class SourceIndependenceScorer:
    """
    Detects wire-service laundering and citation echo chambers.

    When many sources share nearly identical vocabulary, they are almost
    certainly all re-publishing the same wire report or press release.
    High source count with low independence = one claim amplified, not confirmed.
    """

    @staticmethod
    def _token_set(text: str) -> set[str]:
        return set(re.findall(r"[a-z0-9]{4,}", text.lower()))

    @classmethod
    def jaccard(cls, a: str, b: str) -> float:
        sa, sb = cls._token_set(a), cls._token_set(b)
        if not sa or not sb:
            return 0.0
        return len(sa & sb) / len(sa | sb)

    @classmethod
    def score(cls, snippets: list[str]) -> dict[str, Any]:
        """
        Returns independence_score 0.0–1.0 (1.0 = every source is distinct).
        Also flags clone pairs (Jaccard ≥ 0.5).
        """
        if len(snippets) < 2:
            return {"independence_score": 1.0, "clone_pairs": 0, "total_pairs": 0, "warning": ""}

        clone_pairs = 0
        total_pairs = 0
        for i in range(len(snippets)):
            for j in range(i + 1, len(snippets)):
                total_pairs += 1
                if cls.jaccard(snippets[i], snippets[j]) >= 0.5:
                    clone_pairs += 1

        independence = round(1.0 - (clone_pairs / max(1, total_pairs)), 3)
        if independence < 0.5:
            warning = (
                f"Wire laundering likely: {clone_pairs}/{total_pairs} source pairs share >50% vocabulary. "
                "Multiple outlets may be republishing a single press release or wire report."
            )
        elif independence < 0.7:
            warning = (
                f"Echo chamber signal: {clone_pairs}/{total_pairs} source pairs are near-duplicate. "
                "Treat these as one confirmed source, not many."
            )
        else:
            warning = ""

        return {
            "independence_score": independence,
            "clone_pairs": clone_pairs,
            "total_pairs": total_pairs,
            "warning": warning,
        }


class ConsensusAlarmSystem:
    """
    'Too-perfect consensus' detector.

    When all sources agree in near-identical language AND no tier-1 anchor
    is present, the result set is suspicious. Common patterns:
      - Coordinated PR campaigns
      - Astroturf / influencer blast
      - Wire service regurgitation with no original reporting
      - SEO content farms all copying the same source

    A real story backed by real evidence typically produces varied reporting:
    different angles, different wording, some disagreement.
    Uniformity is a red flag, not a quality signal.
    """

    @classmethod
    def evaluate(cls, sources: list[dict[str, Any]]) -> dict[str, Any]:
        if len(sources) < 3:
            return {"alarm": False, "uniformity_ratio": 0.0, "has_tier1_anchor": False, "reason": ""}

        snippets = [str(s.get("snippet", "")) for s in sources if str(s.get("snippet", "")).strip()]
        tiers = [str(s.get("source_tier", "tier3")) for s in sources]

        if len(snippets) < 3:
            return {"alarm": False, "uniformity_ratio": 0.0, "has_tier1_anchor": any(t == "tier1" for t in tiers), "reason": ""}

        has_tier1 = any(t == "tier1" for t in tiers)
        total_pairs = 0
        high_sim_pairs = 0
        for i in range(len(snippets)):
            for j in range(i + 1, len(snippets)):
                total_pairs += 1
                if SourceIndependenceScorer.jaccard(snippets[i], snippets[j]) >= 0.4:
                    high_sim_pairs += 1

        uniformity_ratio = round(high_sim_pairs / max(1, total_pairs), 3)
        alarm = uniformity_ratio >= 0.6 and not has_tier1
        reason = ""
        if alarm:
            reason = (
                f"Too-perfect consensus: {high_sim_pairs}/{total_pairs} source pairs share similar vocabulary, "
                f"no tier-1 anchor present. Possible PR campaign, coordinated messaging, or wire regurgitation. "
                f"Treat with skepticism — look for an original primary source."
            )

        return {
            "alarm": alarm,
            "uniformity_ratio": uniformity_ratio,
            "has_tier1_anchor": has_tier1,
            "reason": reason,
        }


class CrossDomainContradictionDetector:
    """
    Extends numeric conflict detection to directional / semantic contradictions.

    Looks for cases where sources in the same result set make opposing claims
    about the same subject — one says something increases, another says it falls;
    one says X succeeded, another says X failed.

    This catches: stock contradictions, election result disputes, scientific
    finding reversals, sports outcome disagreements, and policy claim fights.
    """

    POSITIVE_SIGNALS = {
        "increases", "rises", "grew", "grows", "improves", "confirms", "proved",
        "succeeds", "wins", "gained", "gains", "advances", "leads", "outperforms",
        "surges", "rallies", "recovers", "beats", "tops",
    }
    NEGATIVE_SIGNALS = {
        "decreases", "falls", "fell", "shrinks", "worsens", "denies", "disproves",
        "fails", "loses", "dropped", "drops", "retreats", "trails", "underperforms",
        "plunges", "collapses", "misses", "loses", "declines",
    }

    @classmethod
    def detect(cls, sources: list[dict[str, Any]]) -> list[dict[str, Any]]:
        subject_signals: dict[str, list[tuple[str, str, str]]] = {}

        for src in sources:
            domain = str(src.get("source_domain", src.get("domain", "")))
            tier = str(src.get("source_tier", "tier3"))
            text = f"{src.get('title', '')} {src.get('snippet', '')}".lower()
            words = text.split()
            for i, word in enumerate(words):
                clean = word.strip(".,;:!?\"'()")
                if clean not in cls.POSITIVE_SIGNALS and clean not in cls.NEGATIVE_SIGNALS:
                    continue
                subject = " ".join(words[max(0, i - 2):i]).strip(".,;:!?\"'()")
                if not subject or len(subject) < 4:
                    continue
                direction = "positive" if clean in cls.POSITIVE_SIGNALS else "negative"
                subject_signals.setdefault(subject, []).append((domain, tier, direction))

        contradictions: list[dict[str, Any]] = []
        for subject, signals in subject_signals.items():
            pos = [s[0] for s in signals if s[2] == "positive"]
            neg = [s[0] for s in signals if s[2] == "negative"]
            if pos and neg and set(pos) != set(neg):
                contradictions.append({
                    "subject": subject,
                    "positive_sources": pos[:3],
                    "negative_sources": neg[:3],
                    "note": (
                        f"Directional contradiction on '{subject}': "
                        f"{len(pos)} source(s) positive, {len(neg)} negative."
                    ),
                })

        return contradictions[:5]
