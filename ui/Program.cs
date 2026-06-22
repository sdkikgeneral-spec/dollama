using Dollama.Ui.Components;
using Dollama.Ui.Services;
using Dollama.Ui.Telemetry;

var builder = WebApplication.CreateBuilder(args);

// Blazor Server (インタラクティブサーバーコンポーネント)
builder.Services.AddRazorComponents()
    .AddInteractiveServerComponents();

// SignalR (テレメトリ push)
builder.Services.AddSignalR();

// C++ 生成サーバーを叩く型付き HttpClient。BaseUrl は appsettings の Dollama:BaseUrl。
var baseUrl = builder.Configuration["Dollama:BaseUrl"] ?? "http://127.0.0.1:8080";
builder.Services.AddHttpClient<DollamaClient>(c =>
{
    c.BaseAddress = new Uri(baseUrl);
    c.Timeout = TimeSpan.FromMinutes(5); // 本生成は 84s 規模になり得るため長め
});

// 生成中フラグ (Broadcaster と Generate.razor で共有) と テレメトリ常駐サービス
builder.Services.AddSingleton<GenerationActivity>();
builder.Services.AddHostedService<TelemetryBroadcaster>();

// プリセット永続化ストア (ui/data/presets.json)
builder.Services.AddSingleton<Dollama.Ui.Services.PresetStore>();

var app = builder.Build();

if (!app.Environment.IsDevelopment())
{
    app.UseExceptionHandler("/Error", createScopeForErrors: true);
    app.UseHsts();
}
app.UseStatusCodePagesWithReExecute("/not-found", createScopeForStatusCodePages: true);

app.UseAntiforgery();

app.MapStaticAssets();
app.MapHub<TelemetryHub>("/hubs/telemetry");
app.MapRazorComponents<App>()
    .AddInteractiveServerRenderMode();

app.Run();
