# Stack patterns — Avalonia (.NET desktop)

## Directory structure

```
{{project_name}}/
  Models/            # Plain C# classes — data only, no UI concerns
  ViewModels/        # MVVM ViewModels — inherit ViewModelBase, use ReactiveUI or CommunityToolkit
  Views/             # AXAML + code-behind (.axaml + .axaml.cs)
  Services/          # Business logic, file I/O, async operations
  Assets/            # Icons, images, fonts
  App.axaml          # Application entry, resource dictionaries
  App.axaml.cs       # DI container setup, service registration
  Program.cs         # AppBuilder entry point
{{project_name}}.Tests/
  ViewModels/
  Services/
```

## MVVM pattern — ViewModels are the core

```csharp
// ViewModels/MainViewModel.cs
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;

public partial class MainViewModel : ObservableObject
{
    [ObservableProperty]
    private string _title = "App";

    [ObservableProperty]
    private ObservableCollection<ItemViewModel> _items = new();

    [RelayCommand]
    private async Task LoadItemsAsync()
    {
        var items = await _itemService.GetAllAsync();
        Items = new ObservableCollection<ItemViewModel>(items.Select(i => new ItemViewModel(i)));
    }

    public MainViewModel(IItemService itemService)
    {
        _itemService = itemService;
    }
    private readonly IItemService _itemService;
}
```

## View pattern — minimal code-behind

```xml
<!-- Views/MainView.axaml -->
<UserControl xmlns="https://github.com/avaloniaui"
             x:Class="{{project_name}}.Views.MainView">
  <StackPanel>
    <TextBlock Text="{Binding Title}" />
    <Button Command="{Binding LoadItemsCommand}" Content="Load" />
    <ListBox ItemsSource="{Binding Items}" />
  </StackPanel>
</UserControl>
```

```csharp
// Views/MainView.axaml.cs — keep this nearly empty
public partial class MainView : UserControl
{
    public MainView() => InitializeComponent();
}
```

Logic belongs in ViewModels, not code-behind.

## Dependency injection in App.axaml.cs

```csharp
// App.axaml.cs
public partial class App : Application
{
    public override void OnFrameworkInitializationCompleted()
    {
        var services = new ServiceCollection();
        services.AddSingleton<IItemService, ItemService>();
        services.AddTransient<MainViewModel>();
        var provider = services.BuildServiceProvider();
        if (ApplicationLifetime is IClassicDesktopStyleApplicationLifetime desktop)
            desktop.MainWindow = new MainWindow { DataContext = provider.GetRequiredService<MainViewModel>() };
        base.OnFrameworkInitializationCompleted();
    }
}
```

## Service pattern

```csharp
// Services/IItemService.cs
public interface IItemService
{
    Task<IEnumerable<Item>> GetAllAsync();
    Task<Item> CreateAsync(ItemCreateDto dto);
}

// Services/ItemService.cs — implements IItemService, no UI dependencies
```

## Naming conventions

- ViewModels: `{View}ViewModel` — `MainViewModel`, `ItemDetailViewModel`
- Views: `{Name}View` or `{Name}Window` — `MainView`, `SettingsWindow`
- Services: `I{Name}Service` interface + `{Name}Service` implementation
- Commands: `{Verb}Command` or `{Verb}AsyncCommand` — `LoadItemsCommand`
- Observable properties: PascalCase — `Title`, `Items`, `IsLoading`

## Common mistakes to avoid

- Do NOT put logic in code-behind (`.axaml.cs`) — use ViewModels
- Do NOT use `Thread.Sleep` — use `async/await` throughout
- Do NOT bind directly to Model objects — wrap in ItemViewModel if they need UI state
- Do NOT create services inside ViewModels — inject via constructor
- Do NOT use `Dispatcher.UIThread.Post` unless you're updating from a background thread
