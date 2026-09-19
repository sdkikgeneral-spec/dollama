using System.Text.Json;
using System.Text.Json.Serialization;

namespace Dollama.Ui.Services;

// 複数 dollama エンドポイント (C++ 生成サーバー) の切替を担うシングルトン。
//
// ソースは 2 つ:
//   ① appsettings の "Dollama:Endpoints" (無ければ "Dollama:BaseUrl"・それも無ければ既定値) — FromConfig=true
//   ② ui/data/endpoints.json (UI の「＋」で追加したもの) — FromConfig=false
// 同じ Name が両方にあれば URL は JSON が優先する (ユーザーが上書きした形)。
// FromConfig な項目は Remove できない (設定ファイルが正典のため)。
//
// FavoriteTagStore と同じ流儀: _gate ロック・壊れ JSON は空扱い・一時ファイル→File.Move で
// アトミック書込。
public sealed class EndpointRegistry
{
    public sealed record Endpoint(string Name, string Url, bool FromConfig);

    private readonly string _path;
    private readonly object _gate = new();
    private static readonly JsonSerializerOptions JsonOpts = new()
    {
        WriteIndented = true,
        Encoder = System.Text.Encodings.Web.JavaScriptEncoder.UnsafeRelaxedJsonEscaping,
    };

    private readonly List<Endpoint> _configEndpoints;
    private List<Endpoint> _merged;
    private string _currentName;

    public EndpointRegistry(IConfiguration config, IHostEnvironment env)
    {
        var dataDir = Path.Combine(env.ContentRootPath, "data");
        _path = Path.Combine(dataDir, "endpoints.json");

        _configEndpoints = LoadConfigEndpoints(config);
        _merged = Merge(_configEndpoints, LoadPersisted());
        _currentName = _merged.Count > 0 ? _merged[0].Name : "local";
    }

    // 現在有効な全エンドポイント (設定由来 + JSON 由来のマージ済みビュー)。
    public IReadOnlyList<Endpoint> All
    {
        get
        {
            lock (_gate)
            {
                return _merged;
            }
        }
    }

    // 現在選択中のエンドポイント。All が空になることは無い (最低 1 件は設定由来で必ず存在する)。
    public Endpoint Current
    {
        get
        {
            lock (_gate)
            {
                return _merged.FirstOrDefault(e => e.Name == _currentName) ?? _merged[0];
            }
        }
    }

    public event Action? Changed;

    // 名前で選択を切り替える。見つからなければ何もせず false。
    public bool Select(string name)
    {
        lock (_gate)
        {
            if (!_merged.Any(e => e.Name == name))
            {
                return false;
            }
            _currentName = name;
        }
        Changed?.Invoke();
        return true;
    }

    // JSON 由来のエンドポイントを追加 (同名なら URL を上書き)。
    // 空 name/url・絶対 URL でない・scheme が http/https 以外は失敗として false。
    public bool Add(string name, string url)
    {
        name = (name ?? "").Trim();
        url = (url ?? "").Trim();
        if (name.Length == 0 || url.Length == 0)
        {
            return false;
        }
        if (!Uri.TryCreate(url, UriKind.Absolute, out var parsed) ||
            (parsed.Scheme != Uri.UriSchemeHttp && parsed.Scheme != Uri.UriSchemeHttps))
        {
            return false;
        }

        lock (_gate)
        {
            var persisted = LoadPersisted();
            persisted.RemoveAll(e => string.Equals(e.Name, name, StringComparison.Ordinal));
            persisted.Add(new PersistedEndpoint(name, url));
            Persist(persisted);
            _merged = Merge(_configEndpoints, persisted);
        }
        Changed?.Invoke();
        return true;
    }

    // JSON 由来のエンドポイントを削除する。FromConfig (設定由来) は削除不可で false。
    // 見つからない場合も false。
    public bool Remove(string name)
    {
        lock (_gate)
        {
            var target = _merged.FirstOrDefault(e => e.Name == name);
            if (target is null || target.FromConfig)
            {
                return false;
            }

            var persisted = LoadPersisted();
            var removed = persisted.RemoveAll(e => string.Equals(e.Name, name, StringComparison.Ordinal)) > 0;
            if (!removed)
            {
                return false;
            }
            Persist(persisted);
            _merged = Merge(_configEndpoints, persisted);
            if (_currentName == name)
            {
                _currentName = _merged.Count > 0 ? _merged[0].Name : _currentName;
            }
        }
        Changed?.Invoke();
        return true;
    }

    // --- 内部 ---

    private static List<Endpoint> LoadConfigEndpoints(IConfiguration config)
    {
        var section = config.GetSection("Dollama:Endpoints");
        var fromSection = section.Get<List<ConfigEndpointDto>>();
        if (fromSection is { Count: > 0 })
        {
            var filtered = fromSection
                .Where(e => !string.IsNullOrWhiteSpace(e.Name) && !string.IsNullOrWhiteSpace(e.Url))
                .Select(e => new Endpoint(e.Name!.Trim(), e.Url!.Trim(), FromConfig: true))
                .ToList();
            // 項目はあるが Name/Url が全部空白だとここが空になる。
            // 空リストのまま返すと _merged が空になり Current の [0] アクセスで例外になるため、
            // BaseUrl フォールバックへ落とす (下の共通経路へ続ける)。
            if (filtered.Count > 0)
            {
                return filtered;
            }
        }

        var baseUrl = config["Dollama:BaseUrl"];
        if (string.IsNullOrWhiteSpace(baseUrl))
        {
            baseUrl = "http://127.0.0.1:8080";
        }
        return new List<Endpoint> { new("local", baseUrl.Trim(), FromConfig: true) };
    }

    // config (FromConfig=true) を土台に、JSON 由来 (persisted) を Name で上書きマージする。
    // Name が config と衝突する persisted 項目は「URL は JSON 優先・削除不可は維持」= FromConfig=true のまま。
    // 衝突しない persisted 項目は純粋な追加 (FromConfig=false・削除可)。
    private static List<Endpoint> Merge(List<Endpoint> configEndpoints, List<PersistedEndpoint> persisted)
    {
        var map = new Dictionary<string, Endpoint>(StringComparer.Ordinal);
        var order = new List<string>();
        foreach (var e in configEndpoints)
        {
            if (!map.ContainsKey(e.Name))
            {
                order.Add(e.Name);
            }
            map[e.Name] = e;
        }
        foreach (var p in persisted)
        {
            var fromConfig = map.ContainsKey(p.Name) && map[p.Name].FromConfig;
            if (!map.ContainsKey(p.Name))
            {
                order.Add(p.Name);
            }
            map[p.Name] = new Endpoint(p.Name, p.Url, fromConfig);
        }
        return order.Select(name => map[name]).ToList();
    }

    private List<PersistedEndpoint> LoadPersisted()
    {
        try
        {
            if (!File.Exists(_path))
            {
                return new List<PersistedEndpoint>();
            }
            var json = File.ReadAllText(_path);
            return JsonSerializer.Deserialize<List<PersistedEndpoint>>(json, JsonOpts)
                   ?? new List<PersistedEndpoint>();
        }
        catch
        {
            // 壊れ JSON 等は空扱い (起動・操作を止めない)。
            return new List<PersistedEndpoint>();
        }
    }

    private void Persist(List<PersistedEndpoint> list)
    {
        var dir = Path.GetDirectoryName(_path)!;
        Directory.CreateDirectory(dir);
        var json = JsonSerializer.Serialize(list, JsonOpts);
        var tmp = _path + ".tmp";
        File.WriteAllText(tmp, json);
        File.Move(tmp, _path, overwrite: true);
    }

    // appsettings "Dollama:Endpoints" バインド用。
    private sealed class ConfigEndpointDto
    {
        public string? Name { get; set; }
        public string? Url { get; set; }
    }

    // data/endpoints.json の永続化形式。
    private sealed record PersistedEndpoint(
        [property: JsonPropertyName("name")] string Name,
        [property: JsonPropertyName("url")] string Url);
}
