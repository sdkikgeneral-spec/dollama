using System.Text.RegularExpressions;
using Xunit;

namespace Dollama.Ui.Tests;

// P3-5 (直近生成のミニ履歴) の **結線側** の回帰止め。
// docs/ui-brushup-plan.md §5 「P3 バッチ F 実装メモ」参照。
//
// 純ロジック (並び替え・容量・選択判定) は Services/PreviewHistory.cs + PreviewHistoryTests が
// 押さえている。ここが見るのは **razor と app.css の結線**で、どれも
// 「壊れても画面はそれらしく描画されてしまう」種類のものだけ。
//
// ★ F-1 が本ファイルの最重要項目。
//   `_imageData` (絵) / `_imageBytes` (プリセット保存のサムネ元) / `_lastMode` (モードバッジ) /
//   `_downloadName` (PNG 保存のファイル名) は **同じ 1 回の生成を指す 1 組**で、
//   分離代入を許すと「古いサムネを押したら絵だけ戻ってバッジは本番のまま・
//   PNG 保存が別の生成のファイル名」という **F 最大の事故**が起きる。
//   目視ではまず気づけない (絵は正しく戻るので成功したように見える) ので、
//   「4 つへの代入は private メソッド ShowResult の中にちょうど 1 つずつ」という
//   **構文レベルの制約**として固定する。
//
// ★ 検査対象一覧 (PL 指定の F-1〜F-8)
//   F-1 [funnel]       4 フィールドへの代入は ShowResult の中だけ・各 1 箇所
//   F-2 [history-add]  PreviewHistory.Add は try の成功経路に 1 箇所だけ (catch/finally に無い)
//   F-3 [history-type] _history は IReadOnlyList で、その場 Mutate しない
//   F-4 [busy-guard]   クリックハンドラ冒頭の _busy ガード + markup 側の disabled
//   F-5 [persist]      履歴経路に永続化 (File. / PresetStore / JSRuntime / localStorage) が無い
//   F-6 [breadth]      履歴 markup に削除ボタン相当・draggable・サムネ毎の保存導線が無い
//   F-7 [sizing]       サムネ列が P1-6 の実効 px 表を 1 行も動かさない
//   F-8 [cap]          E の頭打ち N が「104 + サムネ列の高さ」へ更新されている
//
// bUnit は入れない (依存追加ゼロの既存方針どおり razor / CSS をテキストとして読む)。
// CSS パーサ (Blocks / TopLevelCss / MediaBlocks / DeclarationValue …) と razor のタグ解析
// (ElementTags / ClassAttribute / SplitClassTokens) は AppCssTokenTests の partial から共有する。
public sealed partial class AppCssTokenTests
{
    // funnel が束ねる 4 フィールド。**この集合が P3-5 の一貫性の単位**。
    private static readonly string[] FunnelFields =
    {
        "_imageData", "_imageBytes", "_lastMode", "_downloadName",
    };

    // 4 フィールド × 各 1 箇所。件数を定数で固定してあるので、黙って 1 つ増やすと赤くなる。
    private const int ExpectedFunnelAssignments = 4;

    // P3-4 (バッチ E) が置いたプレビュー枠の頭打ちの基準値。
    // トップバー (~39) + .main の上下 padding (32) + .canvas の枠と上下 padding (34)。
    private const int PreviewCapBasePx = 104;

    // 履歴経路に出てはいけない永続化の入口 (F-5)。
    private static readonly string[] PersistenceMarkers =
    {
        "File.", "Directory.", "PresetStore", "JSRuntime", "localStorage", "sessionStorage",
    };

    // 履歴 markup に出てはいけないもの (F-6)。境界を越える手が入った合図。
    //   削除ボタン相当 (× / btn-icon / Delete) / 並べ替え (draggable) / ピン留め (Pin) /
    //   サムネごとのダウンロード導線 (download = PNG 保存は画面に 1 つのまま) /
    //   件数の切り詰め (Take / Skip = 枚数の規則は PreviewHistory だけが持つ)
    private static readonly string[] BreadthMarkers =
    {
        "draggable", "download", "btn-icon", "Delete", "Remove", "Pin", "Take(", "Skip(",
    };

    // ────────────────────────────────────────────────
    // (20) 本番判定 — 出荷している Generate.razor / app.css がすべての規則を満たす
    // ────────────────────────────────────────────────

    [Fact]
    public void PreviewHistoryWiring_SatisfiesEveryRule()
    {
        var problems = FunnelProblems(File.ReadAllText(GenerateRazorPath));
        Assert.True(problems.Count == 0, "ミニ履歴の結線規則違反: " + string.Join(" / ", problems));
    }

    [Fact]
    public void ThumbStrip_SatisfiesEveryRule()
    {
        var problems = ThumbStripProblems(File.ReadAllText(AppCssPath));
        Assert.True(problems.Count == 0, "サムネ列の規則違反: " + string.Join(" / ", problems));
    }

    // ────────────────────────────────────────────────
    // (21) F-1 funnel — 4 フィールドは 1 組でしか動かない
    // ────────────────────────────────────────────────

    [Theory]
    [InlineData("_imageData")]
    [InlineData("_imageBytes")]
    [InlineData("_lastMode")]
    [InlineData("_downloadName")]
    public void FunnelField_IsAssignedExactlyOnceAndOnlyInsideShowResult(string field)
    {
        var razor = File.ReadAllText(GenerateRazorPath);
        var code = MaskCs(RazorCodeBlock(razor)!);
        var show = MethodBodyRange(code, "private void ShowResult(");

        Assert.True(show is not null, "private void ShowResult(...) が @code に無い");

        var all = Assignments(code, field);

        Assert.True(all.Count == 1, $"{field} への代入は 1 箇所であること (実際 {all.Count} 箇所)");
        Assert.InRange(all[0], show!.Value.Start, show.Value.End);

        // markup 側 (@{ … } を含む) からも代入していないこと
        Assert.Empty(Assignments(StripRazorComments(RazorMarkup(razor)), field));
    }

    [Fact]
    public void Funnel_HoldsExactlyTheFourFieldsThatMustAgree()
    {
        var code = MaskCs(RazorCodeBlock(File.ReadAllText(GenerateRazorPath))!);
        var show = MethodBodyRange(code, "private void ShowResult(")!.Value;
        var body = code[show.Start..show.End];

        // 4 つとも ShowResult の中に居る (どれか 1 つが抜けると「1 組」でなくなる)
        Assert.Equal(
            ExpectedFunnelAssignments,
            FunnelFields.Sum(f => Assignments(body, f).Count));

        // 選択中サムネの印も同じ 1 箇所で更新する (絵とサムネの選択がずれない)
        Assert.Single(Assignments(body, "_selectedId"));
    }

    // base64 → 生バイトの decode は 1 箇所だけ (履歴は base64 しか持たない設計の実効装置)。
    // ここが 2 箇所になると「履歴にも byte[] を持たせた」等の設計逸脱が始まっている。
    [Fact]
    public void Decode_HappensOnlyInsideShowResult()
    {
        var code = MaskCs(RazorCodeBlock(File.ReadAllText(GenerateRazorPath))!);
        var show = MethodBodyRange(code, "private void ShowResult(")!.Value;

        var hits = Occurrences(code, "Convert.FromBase64String");

        Assert.Single(hits);
        Assert.InRange(hits[0], show.Start, show.End);
    }

    // 呼び出し元は 2 つだけ = 生成成功時と履歴クリック時。
    [Fact]
    public void ShowResult_IsCalledFromExactlyTwoPlaces()
    {
        var code = MaskCs(RazorCodeBlock(File.ReadAllText(GenerateRazorPath))!);

        // 宣言 1 + 呼び出し 2
        Assert.Equal(3, Occurrences(code, "ShowResult(").Count);

        var gen = MethodBodyRange(code, "private async Task GenerateAsync(")!.Value;
        var select = MethodBodyRange(code, "private void SelectHistory(")!.Value;

        Assert.Single(Occurrences(code[gen.Start..gen.End], "ShowResult("));
        Assert.Single(Occurrences(code[select.Start..select.End], "ShowResult("));
    }

    // ────────────────────────────────────────────────
    // (22) F-2 履歴に積むのは成功したときだけ
    // ────────────────────────────────────────────────

    [Fact]
    public void HistoryAdd_LivesOnlyOnTheSuccessPathOfTheTryBlock()
    {
        var code = MaskCs(RazorCodeBlock(File.ReadAllText(GenerateRazorPath))!);
        var gen = MethodBodyRange(code, "private async Task GenerateAsync(")!.Value;

        var hits = Occurrences(code, "PreviewHistory.Add(");
        Assert.Single(hits);

        var (tryStart, tryEnd) = KeywordBlock(code, gen, "try")!.Value;
        Assert.InRange(hits[0], tryStart, tryEnd);

        // catch / finally には 1 つも無い (失敗した生成がサムネ列に並ばない)
        foreach (var (start, end) in KeywordBlocks(code, gen, "catch").Concat(KeywordBlocks(code, gen, "finally")))
        {
            Assert.DoesNotContain("PreviewHistory.Add(", code[start..end]);
        }
    }

    // ────────────────────────────────────────────────
    // (23) F-3 その場 Mutate をしない (再描画が走らない本プロジェクト頻出バグ)
    // ────────────────────────────────────────────────

    [Fact]
    public void History_IsAnImmutableListReassignedThroughPreviewHistory()
    {
        var code = MaskCs(RazorCodeBlock(File.ReadAllText(GenerateRazorPath))!);

        // 宣言は 1 つだけで、型は IReadOnlyList (List にすると Mutate できてしまう)
        var declarations = Regex.Matches(code, @"private\s+(\S+)\s+_history\b");
        Assert.Single(declarations);
        Assert.Equal("IReadOnlyList<PreviewItem>", declarations[0].Groups[1].Value);

        foreach (var mutator in new[] { "_history.Add(", "_history.Insert(", "_history.RemoveAt(", "_history.Remove(", "_history.Clear(" })
        {
            Assert.DoesNotContain(mutator, code);
        }

        // 代入は「初期化 (空)」と「PreviewHistory.Add の戻り値」の 2 通りだけ
        foreach (var rhs in AssignedExpressions(code, "_history"))
        {
            Assert.True(
                rhs.StartsWith("Array.Empty<", StringComparison.Ordinal)
                || rhs.StartsWith("PreviewHistory.Add(", StringComparison.Ordinal),
                $"_history へ想定外の代入: {rhs}");
        }
    }

    // ────────────────────────────────────────────────
    // (24) F-4 生成中はサムネを押せない
    // ────────────────────────────────────────────────

    [Fact]
    public void ClickHandler_StartsWithTheBusyGuard()
    {
        var code = RazorCodeBlock(File.ReadAllText(GenerateRazorPath))!;
        var masked = MaskCs(code);
        var range = MethodBodyRange(masked, "private void SelectHistory(")!.Value;

        // コメントを落として空白を畳んだ「最初の文」がガードであること
        var body = Normalize(masked[range.Start..range.End]);
        Assert.StartsWith("if (_busy) { return; }", body, StringComparison.Ordinal);
    }

    [Fact]
    public void ThumbButtons_AreDisabledWhileGenerating()
    {
        var region = HistoryMarkupRegion(StripRazorComments(RazorMarkup(File.ReadAllText(GenerateRazorPath))))!;

        var buttons = ElementTags(region)
            .Where(t => SplitClassTokens(ClassAttribute(t) ?? "").Static.Contains("thumb", StringComparer.Ordinal))
            .ToList();

        Assert.Single(buttons);
        Assert.StartsWith("<button", buttons[0]);
        Assert.Contains("disabled=", buttons[0]);
        Assert.Contains("_busy", buttons[0]);

        // 選択中の印は純クラスの判定を通す (razor 側で自前比較しない)
        Assert.Contains("PreviewHistory.IsSelected(", region);
        Assert.Contains("on", SplitClassTokens(ClassAttribute(buttons[0]) ?? "").Dynamic);
    }

    // ────────────────────────────────────────────────
    // (25) F-5 / F-6 breadth の境界
    // ────────────────────────────────────────────────

    [Fact]
    public void HistoryPath_HasNoPersistenceAtAll()
    {
        var razor = File.ReadAllText(GenerateRazorPath);
        var code = MaskCs(RazorCodeBlock(razor)!);
        var region = HistoryMarkupRegion(StripRazorComments(RazorMarkup(razor)))!;

        var scopes = new List<string> { region };
        foreach (var signature in new[] { "private void ShowResult(", "private void SelectHistory(" })
        {
            var r = MethodBodyRange(code, signature)!.Value;
            scopes.Add(code[r.Start..r.End]);
        }

        // 履歴の実体を持つ純クラス側も同じ制約 (ここに永続化が生えたら breadth 逸脱)。
        // ★ コメントはマスクしてから見る — 「localStorage はやらない」と**書いてある**のを
        //   永続化と読み違えないため (実際この検査を書いた時に 1 度踏んだ)。
        scopes.Add(MaskCs(File.ReadAllText(Path.Combine(RepoRoot(), "ui", "Services", "PreviewHistory.cs"))));

        foreach (var scope in scopes)
        {
            foreach (var marker in PersistenceMarkers)
            {
                Assert.DoesNotContain(marker, scope, StringComparison.Ordinal);
            }
        }
    }

    [Fact]
    public void HistoryMarkup_HasNoDeleteReorderOrPerThumbDownload()
    {
        var region = HistoryMarkupRegion(StripRazorComments(RazorMarkup(File.ReadAllText(GenerateRazorPath))))!;

        foreach (var marker in BreadthMarkers)
        {
            Assert.DoesNotContain(marker, region, StringComparison.Ordinal);
        }

        // PNG 保存の導線は画面に 1 つのまま (履歴の外・右上のボタンだけ)
        var razor = File.ReadAllText(GenerateRazorPath);
        Assert.Single(ElementTags(razor), t => t.Contains("download=", StringComparison.Ordinal));
    }

    // 空なら列ごと出さない (プレースホルダも件数バッジも出さない)。
    [Fact]
    public void ThumbStrip_IsHiddenEntirelyWhenTheHistoryIsEmpty()
    {
        var markup = StripRazorComments(RazorMarkup(File.ReadAllText(GenerateRazorPath)));
        var condition = HistoryMarkupCondition(markup);

        Assert.True(condition is not null, "サムネ列が @if で包まれていない");
        Assert.Contains("_history", condition!);
        Assert.Contains("Count > 0", condition);
    }

    // 履歴はプレビュー枠の外・右ペインの中 (枠に重ねない = 生成物を隠さない)。
    [Fact]
    public void ThumbStrip_SitsInsideTheCanvasButOutsideThePreviewFrame()
    {
        var markup = StripRazorComments(RazorMarkup(File.ReadAllText(GenerateRazorPath)));

        var canvas = markup.IndexOf("class=\"canvas\"", StringComparison.Ordinal);
        var preview = markup.IndexOf("class=\"preview\"", StringComparison.Ordinal);
        var thumbs = markup.IndexOf("class=\"thumbs\"", StringComparison.Ordinal);
        var canvasEnd = markup.IndexOf("</section>", canvas, StringComparison.Ordinal);

        Assert.True(canvas > 0 && preview > canvas, "プレビュー枠が右ペインの中にある");
        Assert.True(thumbs > preview, $"サムネ列はプレビュー枠より後 (実際 preview={preview} / thumbs={thumbs})");
        Assert.True(thumbs < canvasEnd, "サムネ列が右ペイン (.canvas) の外に出ている");

        // ★ プレビュー枠の**中**に入っていないこと。入れ子の深さを数えて枠の閉じ位置を出し、
        //   サムネ列がその後ろに居ることを見る (前に居ると生成画像の上に重なる)。
        var previewTag = ElementTags(markup).First(t => (ClassAttribute(t) ?? "") == "preview");
        var previewStart = markup.IndexOf(previewTag, StringComparison.Ordinal);
        var previewEnd = DivCloseIndex(markup, previewStart);

        Assert.True(previewEnd > 0, "プレビュー枠 (.preview) の閉じタグが見つからない");
        Assert.True(thumbs > previewEnd, $"サムネ列がプレビュー枠の中にある (枠の閉じ={previewEnd} / thumbs={thumbs})");
    }

    // ────────────────────────────────────────────────
    // (26) F-7 P1-6 の実効 px 表を 1 行も動かさない
    // ────────────────────────────────────────────────

    [Fact]
    public void ThumbStrip_StaysOutOfTheP1_6BaselineTables()
    {
        // サムネ列の宣言が (A) 字送り・角丸表にも (A2) 余白表にも 1 行も現れない
        Assert.DoesNotContain(ScaleActual.Value, e => e.Selector.Contains("thumb", StringComparison.Ordinal));
        Assert.DoesNotContain(SpacingActual.Value, e => e.Selector.Contains("thumb", StringComparison.Ordinal));

        // 表そのものの件数もここで直接固定する (バッチ D が凍結した値)
        Assert.Equal(36, ScaleActual.Value.Count(x => x.Property == "font-size"));
        Assert.Equal(29, ScaleActual.Value.Count(x => x.Property == "border-radius"));
        Assert.Equal(65, ScaleActual.Value.Count);
        Assert.Equal(65, SpacingActual.Value.Count);

        // 右ペイン (.canvas) に足したのは flex-direction だけ = 余白も寸法も増えていない
        Assert.Equal(1, SpacingActual.Value.Count(e => e.Selector == ".canvas"));
        Assert.Equal(1, ScaleActual.Value.Count(e => e.Selector == ".canvas"));
        Assert.Equal(1, ScaleActual.Value.Count(e => e.Selector == ".preview"));
    }

    // 縦並びにしないとサムネ列がプレビュー枠の**横**に並ぶ (右ペインが左右 2 分割になる)。
    [Fact]
    public void Canvas_StacksThePreviewAndTheStripVertically()
    {
        var canvas = Blocks(TopLevelCss(File.ReadAllText(AppCssPath)))
            .Single(b => Normalize(b.Selector) == ".canvas");

        Assert.Equal("column", DeclarationValue(canvas.Body, "flex-direction"));
        // 中央寄せは従来どおり (縦並びでは align-items が横中央の担当になる)
        Assert.Equal("center", DeclarationValue(canvas.Body, "align-items"));
        Assert.Equal("center", DeclarationValue(canvas.Body, "justify-content"));
    }

    // ────────────────────────────────────────────────
    // (27) F-8 E の頭打ち N を「104 + サムネ列の高さ」へ更新
    // ────────────────────────────────────────────────

    [Fact]
    public void PreviewCap_GrowsByExactlyTheHeightOfTheThumbStrip()
    {
        var css = File.ReadAllText(AppCssPath);

        var thumbHeight = ThumbOuterHeightPx(css);
        Assert.True(thumbHeight is not null, ".thumb の height が読めない");

        var cap = PreviewCapPx(css);
        Assert.True(cap is not null, ".preview の頭打ち calc(100vh - Npx) が読めない");

        // ★ E の検査意図「デスクトップでプレビュー枠が縦にはみ出さない」は
        //   「枠の実高さ = 100vh - (E の基準) - (サムネ列の高さ)」でこそ保たれる。
        //   サムネ列を足したのに N を据え置くと、枠は列の分だけ縦にはみ出す。
        Assert.Equal(PreviewCapBasePx + thumbHeight!.Value, cap!.Value);
        Assert.Equal(176, cap.Value);

        // 高さに枠を含める前提 (border-box) が崩れると上の足し算が成立しない
        var thumb = Blocks(TopLevelCss(css)).Single(b => Normalize(b.Selector) == ".thumb");
        Assert.Equal("border-box", DeclarationValue(thumb.Body, "box-sizing"));
    }

    // 絶対配置で画像に重ねる案は不採用 (生成物を隠す) — 構造ごと固定する。
    [Fact]
    public void ThumbStrip_IsInFlowAndNeverOverlapsTheImage()
    {
        foreach (var b in ThumbBlocks(File.ReadAllText(AppCssPath)))
        {
            var position = DeclarationValue(b.Body, "position");
            Assert.True(position is null or "static", $"サムネ列が position: {position} を持っている: {b.Selector}");
            Assert.Null(DeclarationValue(b.Body, "z-index"));
        }
    }

    // 選択中は既存の --accent 系トークンだけで表す (新トークンを足さない)。
    [Fact]
    public void SelectedThumb_IsMarkedWithTheExistingAccentToken()
    {
        var selected = ThumbBlocks(File.ReadAllText(AppCssPath))
            .Where(b => SelectorParts(b.Selector).Any(p => p.Contains(".on", StringComparison.Ordinal)))
            .ToList();

        Assert.True(selected.Count == 1, $"選択中サムネの宣言は 1 ブロックであること (実際 {selected.Count} 件)");
        Assert.Equal("var(--accent)", DeclarationValue(selected[0].Body, "border-color"));

        // hover と同詳細度 (0,3,0) の受け皿を併記して、選択中が hover で消えないようにする
        Assert.Contains(".thumb.on:hover", SelectorParts(selected[0].Selector));
    }

    // ══════════════════════════════════════════════════
    //  (28) 検査装置そのものの自己検査 + 変異検査
    //
    //  上の検査は「現行 Generate.razor / app.css に対して緑」なので、スキャナが
    //  壊れていても (@code を見つけられない・コメントを素通しする) 緑のまま素通りしうる。
    //  合成 razor と**本物の Generate.razor に 1 箇所だけ変異を入れたもの**を食わせて、
    //  「噛んでいること」と「壊すと赤くなること」を確かめる。
    // ══════════════════════════════════════════════════

    // 出荷している形を最小構成で写したもの (これを 1 箇所ずつ壊したのが下の変異ケース)。
    // ★ 改行を \n へ揃えてから使う。変異は Replace で入れるので、CRLF/LF の差で
    //   置換が空振りすると「壊したつもりが壊れていない = 変異検査が常に緑」になる
    //   (空振りは下の Assert.NotEqual でも捕まえるが、そもそも起こさない)。
    private static readonly string FunnelRazorShipped = FunnelRazorSource.Replace("\r\n", "\n");

    private const string FunnelRazorSource = """
        <section class="canvas">
            <div class="preview">
                <img src="data:image/png;base64,@_imageData" alt="generated" />
            </div>

            @if (_history.Count > 0)
            {
                <div class="thumbs">
                    @foreach (var item in _history)
                    {
                        <button type="button" @key="item.Id"
                                class="thumb @(PreviewHistory.IsSelected(item, _selectedId) ? "on" : null)"
                                disabled="@_busy"
                                title="@item.Badge"
                                @onclick="@(() => SelectHistory(item.Id))">
                            <img src="data:image/png;base64,@item.Base64" alt="@item.Badge" />
                        </button>
                    }
                </div>
            }
        </section>

        @code {
            private IReadOnlyList<PreviewItem> _history = Array.Empty<PreviewItem>();
            private int _selectedId;
            private int _nextId;

            private async Task GenerateAsync(bool draft = false)
            {
                try
                {
                    var png = await Client.GenerateAsync(req);
                    _connected = true;

                    var item = new PreviewItem(
                        ++_nextId,
                        Convert.ToBase64String(png),
                        PreviewLabel.Build(draft, sendSize),
                        DownloadName.ForPng(DateTime.Now, sendSize),
                        DateTimeOffset.Now);

                    _history = PreviewHistory.Add(_history, item);
                    ShowResult(item);
                }
                catch (GenerationException ex)
                {
                    _error = ex.Message;
                }
                catch (Exception ex)
                {
                    _error = $"接続エラー: {ex.Message}";
                }
                finally
                {
                    _busy = false;
                }
            }

            private void ShowResult(PreviewItem item)
            {
                _imageData = item.Base64;
                _imageBytes = Convert.FromBase64String(item.Base64);
                _lastMode = item.Badge;
                _downloadName = item.DownloadName;
                _selectedId = item.Id;
            }

            private void SelectHistory(int id)
            {
                if (_busy)
                {
                    return;
                }

                var item = PreviewHistory.Find(_history, id);
                if (item is null)
                {
                    return;
                }

                ShowResult(item);
            }
        }
        """;

    [Fact]
    public void FunnelAnalyzer_AcceptsTheShippedShape()
    {
        Assert.Empty(FunnelProblems(FunnelRazorShipped));
    }

    // ★ 変異はすべて **1 行に閉じた文字列** (または正規表現) を宛先にする。
    //   複数行を Replace の宛先にするとインデントと改行コードに依存して空振りしやすい。
    public static IEnumerable<object[]> BrokenFunnelRazorData() => new[]
    {
        // ── F-1 funnel (最大の事故) ──
        new object[]
        {
            "ShowResult の外で _lastMode を書き換える",
            FunnelRazorShipped.Replace(
                "private void ShowResult(",
                "private void LeakMode()\n    {\n        _lastMode = \"本番\";\n    }\n\n    private void ShowResult("),
            "[funnel]",
        },
        new object[]
        {
            "生成側が直接 _imageData も差し替える (バッジだけ古いまま になりうる)",
            FunnelRazorShipped.Replace(
                "_history = PreviewHistory.Add(_history, item);",
                "_history = PreviewHistory.Add(_history, item);\n                    _imageData = item.Base64;"),
            "[funnel]",
        },
        new object[]
        {
            "サムネ側が _downloadName を更新しない (PNG 保存だけ別の生成のまま)",
            FunnelRazorShipped.Replace("_downloadName = item.DownloadName;", ""),
            "[funnel]",
        },
        new object[]
        {
            "decode が 2 箇所に増えた (履歴に byte[] を持たせ始めた合図)",
            FunnelRazorShipped.Replace(
                "var item = new PreviewItem(",
                "var bytes = Convert.FromBase64String(\"\");\n                    var item = new PreviewItem("),
            "[decode]",
        },
        // ── F-2 成功経路 ──
        new object[]
        {
            "失敗した生成も履歴に積む",
            FunnelRazorShipped.Replace(
                "_error = ex.Message;",
                "_error = ex.Message;\n                _history = PreviewHistory.Add(_history, null!);"),
            "[history-add]",
        },
        new object[]
        {
            "履歴に積む処理そのものが消えた",
            FunnelRazorShipped.Replace("_history = PreviewHistory.Add(_history, item);", ""),
            "[history-add]",
        },
        // ── F-3 その場 Mutate ──
        new object[]
        {
            "_history が List になった",
            FunnelRazorShipped.Replace(
                "private IReadOnlyList<PreviewItem> _history = Array.Empty<PreviewItem>();",
                "private List<PreviewItem> _history = new();"),
            "[history-type]",
        },
        new object[]
        {
            "その場 Mutate (再描画が走らない)",
            FunnelRazorShipped.Replace(
                "_history = PreviewHistory.Add(_history, item);",
                "_history.Insert(0, item);"),
            "[history-type]",
        },
        // ── F-4 生成中のクリック ──
        new object[]
        {
            "_busy ガードが無い",
            Regex.Replace(
                FunnelRazorShipped,
                @"(private void SelectHistory\(int id\)\s*\{)\s*if \(_busy\)\s*\{\s*return;\s*\}",
                "$1",
                RegexOptions.Singleline),
            "[busy-guard]",
        },
        new object[]
        {
            "生成中もサムネを押せる (disabled が無い)",
            FunnelRazorShipped.Replace("disabled=\"@_busy\"", ""),
            "[busy-guard]",
        },
        // ── F-5 / F-6 breadth ──
        new object[]
        {
            "履歴をファイルへ書き出し始めた",
            FunnelRazorShipped.Replace(
                "private void SelectHistory(",
                "private void Persist() { File.WriteAllText(\"h.json\", \"\"); }\n\n    private void SelectHistory("),
            "[persist]",
        },
        new object[]
        {
            "サムネごとのダウンロード導線が生えた",
            FunnelRazorShipped.Replace("</button>", "</button>\n<a download=\"x.png\" href=\"#\">保存</a>"),
            "[breadth]",
        },
        new object[]
        {
            "サムネに削除ボタンが生えた",
            FunnelRazorShipped.Replace("</button>", "</button>\n<button class=\"btn btn-icon\">Delete</button>"),
            "[breadth]",
        },
        new object[]
        {
            "空でも列を出す (@if を外した)",
            FunnelRazorShipped.Replace("@if (_history.Count > 0)", "@{"),
            "[empty]",
        },
    };

    [Theory]
    [MemberData(nameof(BrokenFunnelRazorData))]
    public void FunnelAnalyzer_RejectsBrokenShapes(string label, string razor, string code)
    {
        Assert.NotEqual(FunnelRazorShipped, razor);   // Replace が空振りしていないこと

        var problems = FunnelProblems(razor);

        Assert.True(problems.Count > 0, $"変異「{label}」を検出できていない (検査が空振り)");
        Assert.Contains(problems, p => p.StartsWith(code, StringComparison.Ordinal));
    }

    // 合成 razor だけだと「本物の Generate.razor を読めていない」可能性が残るので、
    // **出荷ファイルそのものに 1 箇所だけ変異を入れて**赤くなることも確かめる。
    public static IEnumerable<object[]> MutatedRealRazorData() => new[]
    {
        // ★ PL 指定の実証: ShowResult の外で _lastMode に代入したら赤くなること
        new object[] { "ShowResult の外で _lastMode を書き換える", "lastmode-out", "[funnel]" },
        new object[] { "生成側が直接 _imageBytes を差し替える", "bytes-out", "[funnel]" },
        new object[] { "失敗した生成も履歴に積む", "add-in-catch", "[history-add]" },
        new object[] { "_history が List になった", "mutable-history", "[history-type]" },
        new object[] { "_busy ガードを外す", "no-guard", "[busy-guard]" },
        new object[] { "サムネの disabled を外す", "no-disabled", "[busy-guard]" },
        new object[] { "サムネごとのダウンロード導線", "thumb-download", "[breadth]" },
    };

    [Theory]
    [MemberData(nameof(MutatedRealRazorData))]
    public void RealGenerateRazor_TurnsRedWhenMutated(string label, string mutation, string code)
    {
        var razor = File.ReadAllText(GenerateRazorPath);

        var mutated = mutation switch
        {
            "lastmode-out" => razor.Replace(
                "    private void ShowResult(PreviewItem item)",
                "    private void LeakMode()\n    {\n        _lastMode = \"本番\";\n    }\n\n    private void ShowResult(PreviewItem item)"),
            "bytes-out" => razor.Replace(
                "            ShowResult(item);",
                "            _imageBytes = png;\n            ShowResult(item);"),
            "add-in-catch" => razor.Replace(
                "            _error = ex.Message;",
                "            _error = ex.Message;\n            _history = PreviewHistory.Add(_history, null!);"),
            "mutable-history" => razor.Replace(
                "private IReadOnlyList<PreviewItem> _history = Array.Empty<PreviewItem>();",
                "private List<PreviewItem> _history = new();"),
            "no-guard" => Regex.Replace(
                razor,
                @"(private void SelectHistory\(int id\)\s*\{)\s*if \(_busy\)\s*\{\s*return;\s*\}",
                "$1",
                RegexOptions.Singleline),
            "no-disabled" => razor.Replace("disabled=\"@_busy\"", ""),
            "thumb-download" => razor.Replace(
                "                        </button>",
                "                            <a download=\"x.png\" href=\"#\">保存</a>\n                        </button>"),
            _ => throw new ArgumentOutOfRangeException(nameof(mutation)),
        };

        Assert.NotEqual(razor, mutated);

        var problems = FunnelProblems(mutated);

        Assert.True(problems.Count > 0, $"変異「{label}」を検出できていない (検査が空振り)");
        Assert.Contains(problems, p => p.StartsWith(code, StringComparison.Ordinal));
    }

    // 出荷している CSS の形 (これを 1 箇所ずつ壊したのが下の変異ケース)。
    private const string ThumbCssShipped = """
        .canvas { position: relative; display: flex; flex-direction: column; align-items: center; justify-content: center; padding: 16px; }
        .preview { flex: 1; width: 100%; height: 100%; }
        @media (min-width: 1101px) { .preview { max-width: calc(100vh - 176px); } }
        .thumbs { display: flex; align-items: center; justify-content: center; flex-shrink: 0; }
        .thumb { all: unset; box-sizing: border-box; display: block; width: 72px; height: 72px; background: var(--panel-2); border: 4px solid var(--panel); cursor: pointer; }
        .thumb:hover:not(:disabled) { border-color: var(--accent-border); }
        .thumb.on, .thumb.on:hover { border-color: var(--accent); }
        .thumb:disabled { cursor: default; opacity: 0.45; }
        .thumb img { display: block; width: 100%; height: 100%; object-fit: cover; }
        """;

    [Fact]
    public void ThumbStripAnalyzer_AcceptsTheShippedShape()
    {
        Assert.Empty(ThumbStripProblems(ThumbCssShipped));
    }

    public static IEnumerable<object[]> BrokenThumbCssData() => new[]
    {
        new object[]
        {
            "サムネ列の規則が無い",
            ThumbCssShipped.Replace(".thumbs { display: flex; align-items: center; justify-content: center; flex-shrink: 0; }", ""),
            "[place]",
        },
        new object[]
        {
            "右ペインが横並びのまま (サムネ列が画像の横に出る)",
            ThumbCssShipped.Replace("flex-direction: column; ", ""),
            "[place]",
        },
        new object[]
        {
            "枠の頭打ちがサムネ列の高さを見ていない (枠が縦にはみ出す)",
            ThumbCssShipped.Replace("calc(100vh - 176px)", "calc(100vh - 104px)"),
            "[cap]",
        },
        new object[]
        {
            "サムネの高さを変えたのに頭打ちが追随していない",
            ThumbCssShipped.Replace("height: 72px", "height: 96px"),
            "[cap]",
        },
        new object[]
        {
            "border-box を外した (高さに枠が含まれなくなり足し算が崩れる)",
            ThumbCssShipped.Replace("box-sizing: border-box; ", ""),
            "[cap]",
        },
        new object[]
        {
            "絶対配置で画像に重ねた (生成物を隠す)",
            ThumbCssShipped.Replace(".thumbs { display: flex;", ".thumbs { position: absolute; bottom: 0; display: flex;"),
            "[overlay]",
        },
        new object[]
        {
            "余白 (gap) が生えた = P1-6 の余白表に行が増える",
            ThumbCssShipped.Replace(".thumbs { display: flex;", ".thumbs { gap: var(--sp-2); display: flex;"),
            "[sizing]",
        },
        new object[]
        {
            "角丸が生えた = P1-6 の角丸表に行が増える",
            ThumbCssShipped.Replace("cursor: pointer; }", "cursor: pointer; border-radius: var(--r-sm); }"),
            "[sizing]",
        },
        new object[]
        {
            "UA 既定を落とさなくなった (外寸がブラウザごとにずれる)",
            ThumbCssShipped.Replace("all: unset; ", ""),
            "[reset]",
        },
        new object[]
        {
            "選択中の印が消えた",
            ThumbCssShipped.Replace(".thumb.on, .thumb.on:hover { border-color: var(--accent); }", ""),
            "[select]",
        },
    };

    [Theory]
    [MemberData(nameof(BrokenThumbCssData))]
    public void ThumbStripAnalyzer_RejectsBrokenShapes(string label, string css, string code)
    {
        Assert.NotEqual(ThumbCssShipped, css);

        var problems = ThumbStripProblems(css);

        Assert.True(problems.Count > 0, $"変異「{label}」を検出できていない (検査が空振り)");
        Assert.Contains(problems, p => p.StartsWith(code, StringComparison.Ordinal));
    }

    // ── スキャナ自身の自己検査 ────────────────────────────

    // razor の @code を切り出せること。ここが壊れると上の検査が「1 文字も見ていないのに緑」。
    [Fact]
    public void CodeBlockScanner_ExtractsTheWholeCodeSection()
    {
        const string razor = """
            <div class="a">@_x</div>
            @code {
                private string _x = "{ }";   // 文字列とコメントの { } は数えない
                private void F()
                {
                    _x = "y";
                }
            }
            """;

        var code = RazorCodeBlock(razor);

        Assert.NotNull(code);
        Assert.Contains("private void F()", code);
        Assert.DoesNotContain("<div", code);

        // markup 側は @code の手前まで
        Assert.Contains("<div", RazorMarkup(razor));
        Assert.DoesNotContain("private void F()", RazorMarkup(razor));
    }

    // 文字列・コメントの中身は「コード」として数えない。
    [Fact]
    public void Masker_BlanksOutStringsAndComments()
    {
        const string code = """
            // _lastMode = "x";
            var s = "_lastMode = 1;";
            /* _lastMode = 2; */
            _lastMode = item.Badge;
            """;

        var masked = MaskCs(code);

        // 実コードの 1 箇所だけが代入として見える
        Assert.Single(Assignments(masked, "_lastMode"));
        // マスクは長さを変えない (索引が原文と揃う)
        Assert.Equal(code.Length, masked.Length);
    }

    // 代入と比較を取り違えないこと (== / != / <= を代入と数えたら検査が常時赤になる)。
    [Theory]
    [InlineData("_lastMode = x;", 1)]
    [InlineData("_lastMode= x;", 1)]
    [InlineData("_lastMode ??= x;", 1)]
    [InlineData("_lastMode += x;", 1)]
    [InlineData("_lastMode == x", 0)]
    [InlineData("_lastMode != x", 0)]
    [InlineData("if (_lastMode is not null)", 0)]
    [InlineData("CurrentImage=\"_lastMode\"", 0)]
    [InlineData("x_lastMode = 1;", 0)]
    public void AssignmentScanner_SeparatesAssignmentFromComparison(string snippet, int expected)
    {
        Assert.Equal(expected, Assignments(snippet, "_lastMode").Count);
    }

    // メソッド本体の切り出しが入れ子ブロック (try/catch/if) をまたげること。
    [Fact]
    public void MethodScanner_TakesTheWholeBodyIncludingNestedBlocks()
    {
        const string code = """
            private void A()
            {
                if (x)
                {
                    _a = 1;
                }
            }

            private void B()
            {
                _b = 2;
            }
            """;

        var a = MethodBodyRange(code, "private void A(")!.Value;
        var body = code[a.Start..a.End];

        Assert.Single(Assignments(body, "_a"));
        Assert.Empty(Assignments(body, "_b"));   // 隣のメソッドまで飲み込んでいない
    }

    // try / catch / finally を別々の範囲として取れること (F-2 の土台)。
    [Fact]
    public void KeywordBlockScanner_SeparatesTryFromCatchAndFinally()
    {
        const string code = """
            private void A()
            {
                try
                {
                    _ok = 1;
                }
                catch (Exception ex)
                {
                    _ng = 2;
                }
                finally
                {
                    _fin = 3;
                }
            }
            """;

        var method = MethodBodyRange(code, "private void A(")!.Value;

        var (ts, te) = KeywordBlock(code, method, "try")!.Value;
        Assert.Contains("_ok", code[ts..te]);
        Assert.DoesNotContain("_ng", code[ts..te]);

        Assert.Single(KeywordBlocks(code, method, "catch"));
        Assert.Single(KeywordBlocks(code, method, "finally"));
        Assert.Contains("_ng", KeywordBlocks(code, method, "catch").Select(r => code[r.Start..r.End]).Single());
        Assert.Contains("_fin", KeywordBlocks(code, method, "finally").Select(r => code[r.Start..r.End]).Single());
    }

    // 履歴 markup の切り出しが「サムネ列を含む @if」を選べること。
    [Fact]
    public void MarkupRegionScanner_PicksTheBlockThatHoldsTheStrip()
    {
        const string markup = """
            @if (_busy)
            {
                <div class="preview-overlay"></div>
            }
            @if (_history.Count > 0)
            {
                <div class="thumbs"><button class="thumb"></button></div>
            }
            """;

        var region = HistoryMarkupRegion(markup);

        Assert.NotNull(region);
        Assert.Contains("class=\"thumb\"", region);
        Assert.DoesNotContain("preview-overlay", region);
        Assert.Equal("(_history.Count > 0)", HistoryMarkupCondition(markup));
    }

    // ══════════════════════════════════════════════════
    //  以下 P3-5 用ヘルパー
    // ══════════════════════════════════════════════════

    // ミニ履歴の結線規則を 1 箇所で判定する。
    // 先頭のコードで「どの規則を破ったか」が分かる形 (変異検査が宛先を指定できる)。
    // ★ 例外を投げない — 変異で構造ごと消えた入力も食わされるため。
    private static List<string> FunnelProblems(string razor)
    {
        var problems = new List<string>();

        var markup = StripRazorComments(RazorMarkup(razor));
        var region = HistoryMarkupRegion(markup);
        var raw = RazorCodeBlock(razor);

        if (raw is null)
        {
            problems.Add("[code] @code ブロックが読めない");
            return problems;
        }

        var code = MaskCs(raw);

        // ── F-1 funnel ──
        var show = MethodBodyRange(code, "private void ShowResult(");
        if (show is null)
        {
            problems.Add("[funnel] private void ShowResult(...) が無い (表示中の絵を決める唯一の場所)");
        }

        var total = 0;
        foreach (var field in FunnelFields)
        {
            var hits = Assignments(code, field);
            total += hits.Count;

            if (hits.Count != 1)
            {
                problems.Add($"[funnel] {field} への代入が {hits.Count} 箇所 (1 箇所であること)");
            }

            if (show is not null)
            {
                var outside = hits.Count(i => i < show.Value.Start || i >= show.Value.End);
                if (outside > 0)
                {
                    problems.Add($"[funnel] {field} への代入が ShowResult の外に {outside} 箇所ある " +
                                 "(絵とバッジと保存名が別の生成を指しうる)");
                }
            }

            if (Assignments(markup, field).Count > 0)
            {
                problems.Add($"[funnel] markup から {field} を代入している");
            }
        }

        if (total != ExpectedFunnelAssignments)
        {
            problems.Add($"[funnel] 4 フィールドへの代入が合計 {total} 箇所 ({ExpectedFunnelAssignments} であること)");
        }

        // decode は 1 箇所だけ (履歴が base64 しか持たない設計の実効装置)
        var decodes = Occurrences(code, "Convert.FromBase64String");
        if (decodes.Count != 1)
        {
            problems.Add($"[decode] Convert.FromBase64String が {decodes.Count} 箇所 (1 箇所であること)");
        }
        else if (show is not null && (decodes[0] < show.Value.Start || decodes[0] >= show.Value.End))
        {
            problems.Add("[decode] decode が ShowResult の外にある");
        }

        // ── F-2 履歴に積むのは try の成功経路だけ ──
        var adds = Occurrences(code, "PreviewHistory.Add(");
        var gen = MethodBodyRange(code, "private async Task GenerateAsync(");

        if (adds.Count != 1)
        {
            problems.Add($"[history-add] PreviewHistory.Add の呼び出しが {adds.Count} 箇所 (1 箇所であること)");
        }

        if (gen is null)
        {
            problems.Add("[history-add] GenerateAsync が読めない");
        }
        else
        {
            var tryBlock = KeywordBlock(code, gen.Value, "try");
            if (tryBlock is null)
            {
                problems.Add("[history-add] GenerateAsync に try が無い");
            }
            else if (adds.Count(i => i >= tryBlock.Value.Start && i < tryBlock.Value.End) != 1)
            {
                problems.Add("[history-add] try の成功経路に PreviewHistory.Add がちょうど 1 箇所ない");
            }

            foreach (var (start, end) in KeywordBlocks(code, gen.Value, "catch").Concat(KeywordBlocks(code, gen.Value, "finally")))
            {
                if (adds.Any(i => i >= start && i < end))
                {
                    problems.Add("[history-add] catch / finally から履歴へ積んでいる (失敗した生成が並ぶ)");
                }
            }
        }

        // ── F-3 その場 Mutate ──
        if (!code.Contains("private IReadOnlyList<PreviewItem> _history", StringComparison.Ordinal))
        {
            problems.Add("[history-type] _history が IReadOnlyList<PreviewItem> で宣言されていない");
        }

        foreach (var mutator in new[] { "_history.Add(", "_history.Insert(", "_history.RemoveAt(", "_history.Remove(", "_history.Clear(" })
        {
            if (code.Contains(mutator, StringComparison.Ordinal))
            {
                problems.Add($"[history-type] 履歴をその場で書き換えている ({mutator}) — 再描画が走らない");
            }
        }

        foreach (var rhs in AssignedExpressions(code, "_history"))
        {
            if (!rhs.StartsWith("Array.Empty<", StringComparison.Ordinal)
                && !rhs.StartsWith("PreviewHistory.Add(", StringComparison.Ordinal))
            {
                problems.Add($"[history-type] _history へ想定外の代入: {rhs}");
            }
        }

        // ── F-4 生成中は選べない ──
        var select = MethodBodyRange(code, "private void SelectHistory(");
        if (select is null)
        {
            problems.Add("[busy-guard] private void SelectHistory(...) が無い");
        }
        else if (!Normalize(code[select.Value.Start..select.Value.End])
                     .StartsWith("if (_busy) { return; }", StringComparison.Ordinal))
        {
            problems.Add("[busy-guard] クリックハンドラの冒頭が _busy ガードでない");
        }

        if (region is null)
        {
            problems.Add("[empty] サムネ列を包む @if が見つからない (空でも列が出る)");
        }
        else
        {
            var button = ElementTags(region)
                .FirstOrDefault(t => SplitClassTokens(ClassAttribute(t) ?? "").Static.Contains("thumb", StringComparer.Ordinal));

            if (button is null)
            {
                problems.Add("[busy-guard] サムネのボタン (.thumb) が markup に無い");
            }
            else if (!button.Contains("disabled=", StringComparison.Ordinal) || !button.Contains("_busy", StringComparison.Ordinal))
            {
                problems.Add("[busy-guard] 生成中にサムネが disabled になっていない");
            }

            if (!region.Contains("PreviewHistory.IsSelected(", StringComparison.Ordinal))
            {
                problems.Add("[select] 選択中の判定を純クラス (PreviewHistory.IsSelected) に通していない");
            }

            // ── F-6 breadth ──
            foreach (var marker in BreadthMarkers)
            {
                if (region.Contains(marker, StringComparison.Ordinal))
                {
                    problems.Add($"[breadth] 履歴 markup に {marker} がある (P3-5 の境界外)");
                }
            }
        }

        var condition = HistoryMarkupCondition(markup);
        if (condition is null
            || !condition.Contains("_history", StringComparison.Ordinal)
            || !condition.Contains("Count > 0", StringComparison.Ordinal))
        {
            problems.Add($"[empty] サムネ列の出し分けが「履歴が空でないとき」でない: {condition ?? "(無し)"}");
        }

        // ── F-5 永続化ゼロ ──
        var scopes = new List<string>();
        if (region is not null)
        {
            scopes.Add(region);
        }
        foreach (var range in new[] { show, select })
        {
            if (range is not null)
            {
                scopes.Add(code[range.Value.Start..range.Value.End]);
            }
        }

        foreach (var scope in scopes)
        {
            foreach (var marker in PersistenceMarkers)
            {
                if (scope.Contains(marker, StringComparison.Ordinal))
                {
                    problems.Add($"[persist] 履歴経路に永続化の入口 ({marker}) がある");
                }
            }
        }

        // 履歴のために新しく生えた「永続化っぽいメソッド」も拾う (経路の外に置いても同じこと)
        foreach (var marker in PersistenceMarkers)
        {
            if (code.Contains(marker, StringComparison.Ordinal))
            {
                problems.Add($"[persist] Generate.razor に永続化の入口 ({marker}) がある");
            }
        }

        return problems;
    }

    // サムネ列の CSS 規則を 1 箇所で判定する (同じく例外を投げない)。
    private static List<string> ThumbStripProblems(string css)
    {
        var problems = new List<string>();

        var top = Blocks(TopLevelCss(css));
        var strip = top.Where(b => Normalize(b.Selector) == ".thumbs").ToList();
        var thumb = top.Where(b => Normalize(b.Selector) == ".thumb").ToList();

        // ── 置き場所 ──
        if (strip.Count != 1)
        {
            problems.Add($"[place] .thumbs の宣言が {strip.Count} 件 (1 件であること)");
        }

        if (thumb.Count != 1)
        {
            problems.Add($"[place] .thumb の宣言が {thumb.Count} 件 (1 件であること)");
        }

        var canvas = top.Where(b => Normalize(b.Selector) == ".canvas").ToList();
        if (canvas.Count != 1 || DeclarationValue(canvas[0].Body, "flex-direction") != "column")
        {
            problems.Add("[place] .canvas が縦並び (flex-direction: column) でない " +
                         "— サムネ列がプレビュー枠の横に並ぶ");
        }

        // ── 重ねない / 寸法表を動かさない ──
        foreach (var b in ThumbBlocks(css))
        {
            var position = DeclarationValue(b.Body, "position");
            if (position is not null && position != "static")
            {
                problems.Add($"[overlay] サムネ列が position: {position} を持っている (生成物に重なる)");
            }

            foreach (var d in ParseDeclarations(b.Body).Where(d => IsLayoutForbiddenProperty(d.Property)))
            {
                problems.Add($"[sizing] サムネ列が寸法を持っている: {b.Selector} {{ {d.Property}: {d.Value} }} " +
                             "(P1-6 の実効 px ベースライン表に行が増える)");
            }
        }

        // ── UA 既定のリセット ──
        if (thumb.Count == 1)
        {
            if (DeclarationValue(thumb[0].Body, "all") != "unset")
            {
                problems.Add("[reset] .thumb が button の UA 既定を落としていない (all: unset) " +
                             "— padding: 0 と書くと余白表に行が増えるのでこの形で落とす");
            }

            if (DeclarationValue(thumb[0].Body, "box-sizing") != "border-box")
            {
                problems.Add("[cap] .thumb が border-box でない (height に枠が含まれず頭打ちの足し算が崩れる)");
            }
        }

        // ── 選択中の印 ──
        var selected = ThumbBlocks(css)
            .Where(b => SelectorParts(b.Selector).Any(p => p.Contains(".on", StringComparison.Ordinal)))
            .ToList();

        if (selected.Count != 1 || DeclarationValue(selected[0].Body, "border-color") != "var(--accent)")
        {
            problems.Add("[select] 選択中サムネの印 (border-color: var(--accent)) が 1 ブロックでない");
        }

        // ── E の頭打ちがサムネ列の高さを織り込んでいるか ──
        var height = ThumbOuterHeightPx(css);
        var cap = PreviewCapPx(css);

        if (height is null)
        {
            problems.Add("[cap] .thumb の height が読めない");
        }
        else if (cap is null)
        {
            problems.Add("[cap] .preview の頭打ち calc(100vh - Npx) が読めない");
        }
        else if (cap.Value != PreviewCapBasePx + height.Value)
        {
            problems.Add($"[cap] 頭打ちが {cap.Value}px (E の基準 {PreviewCapBasePx} + サムネ列 {height.Value} = " +
                         $"{PreviewCapBasePx + height.Value} であること — 足りないと枠が縦にはみ出す)");
        }

        return problems;
    }

    // .thumbs / .thumb / .thumb img … サムネ列に属する全ブロック (メディアクエリの中も含む)。
    private static List<CssBlock> ThumbBlocks(string css)
        => Blocks(TopLevelCss(css))
            .Concat(MediaBlocks(css).SelectMany(m => Blocks(m.Inner)))
            .Where(b => SelectorParts(b.Selector).Any(p => p.StartsWith(".thumb", StringComparison.Ordinal)))
            .ToList();

    // サムネ 1 枚の外寸 (border-box なので height がそのまま列の高さ)。
    private static int? ThumbOuterHeightPx(string css)
    {
        var thumb = Blocks(TopLevelCss(css)).Where(b => Normalize(b.Selector) == ".thumb").ToList();
        if (thumb.Count != 1)
        {
            return null;
        }

        var value = DeclarationValue(thumb[0].Body, "height");
        var m = value is null ? Match.Empty : Regex.Match(value, @"^(\d+)px$");
        return m.Success ? int.Parse(m.Groups[1].Value) : null;
    }

    // デスクトップ側のプレビュー枠の頭打ち calc(100vh - Npx) の N。
    private static int? PreviewCapPx(string css)
    {
        var caps = DesktopPreviewQueries(css);
        if (caps.Count != 1)
        {
            return null;
        }

        var block = Blocks(caps[0].Inner)
            .Where(b => SelectorParts(b.Selector).Contains(".preview", StringComparer.Ordinal))
            .ToList();

        if (block.Count != 1)
        {
            return null;
        }

        var value = DeclarationValue(block[0].Body, "max-width");
        var m = value is null ? Match.Empty : Regex.Match(value, @"calc\(\s*100vh\s*-\s*(\d+)px\s*\)");
        return m.Success ? int.Parse(m.Groups[1].Value) : null;
    }

    // ── razor の切り分け ──────────────────────────────

    // @code { … } の中身。無ければ null。
    // ★ 波括弧を数えるのは「@code から先をマスクしたもの」の上で行う。
    //   markup 側の属性値 (入れ子の引用符・ラムダ) を C# として読ませないため。
    private static string? RazorCodeBlock(string razor)
    {
        var i = razor.IndexOf("@code", StringComparison.Ordinal);
        if (i < 0)
        {
            return null;
        }

        var open = razor.IndexOf('{', i);
        if (open < 0)
        {
            return null;
        }

        var end = MatchingBrace(MaskCs(razor[open..]), 0);
        return end < 0 ? null : razor[(open + 1)..(open + end)];
    }

    // @code より手前 (= markup)。
    private static string RazorMarkup(string razor)
    {
        var i = razor.IndexOf("@code", StringComparison.Ordinal);
        return i < 0 ? razor : razor[..i];
    }

    // razor コメント (@* … *@) と HTML コメントを落とす。
    // どちらも描画されないので、breadth の検査で「コメントに書いた語」を拾わないため。
    private static string StripRazorComments(string markup)
    {
        markup = Regex.Replace(markup, @"@\*.*?\*@", "", RegexOptions.Singleline);
        return Regex.Replace(markup, @"<!--.*?-->", "", RegexOptions.Singleline);
    }

    // サムネ列を含む @if ブロックの中身。無ければ null。
    private static string? HistoryMarkupRegion(string markup)
        => HistoryMarkupBlock(markup)?.Body;

    // その @if の条件式 (例 "(_history.Count > 0)")。
    private static string? HistoryMarkupCondition(string markup)
        => HistoryMarkupBlock(markup)?.Condition;

    // markup[openIndex] から始まる <div> に対応する </div> の位置。入れ子を数える。
    private static int DivCloseIndex(string markup, int openIndex)
    {
        var depth = 0;

        for (var i = openIndex; i < markup.Length; i++)
        {
            if (string.CompareOrdinal(markup, i, "<div", 0, 4) == 0)
            {
                depth++;
            }
            else if (string.CompareOrdinal(markup, i, "</div>", 0, 6) == 0)
            {
                depth--;
                if (depth == 0)
                {
                    return i;
                }
            }
        }

        return -1;
    }

    private readonly record struct RazorIfBlock(string Condition, string Body);

    private static RazorIfBlock? HistoryMarkupBlock(string markup)
    {
        foreach (Match m in Regex.Matches(markup, @"@if\b"))
        {
            var open = markup.IndexOf('{', m.Index);
            if (open < 0)
            {
                continue;
            }

            var end = MatchingBrace(markup, open);
            if (end < 0)
            {
                continue;
            }

            var body = markup[(open + 1)..end];
            if (body.Contains("class=\"thumbs\"", StringComparison.Ordinal))
            {
                return new RazorIfBlock(Normalize(markup[(m.Index + "@if".Length)..open]), body);
            }
        }

        return null;
    }

    // ── C# コードの走査 ──────────────────────────────

    // 文字列・文字・コメントの中身を空白へ潰す (長さは変えないので索引が原文と揃う)。
    // これを噛ませてから brace / キーワード / 代入を数えるので、
    // 「コメントに書いた `_lastMode = …`」や「文字列の中の { }」に引っかからない。
    private static string MaskCs(string code)
    {
        var buffer = code.ToCharArray();

        for (var i = 0; i < code.Length; i++)
        {
            var c = code[i];

            if (c == '/' && i + 1 < code.Length && code[i + 1] == '/')
            {
                while (i < code.Length && code[i] != '\n')
                {
                    buffer[i++] = ' ';
                }
                continue;
            }

            if (c == '/' && i + 1 < code.Length && code[i + 1] == '*')
            {
                var close = code.IndexOf("*/", i + 2, StringComparison.Ordinal);
                var stop = close < 0 ? code.Length : close + 2;
                while (i < stop)
                {
                    buffer[i++] = ' ';
                }
                i--;
                continue;
            }

            if (c is '"' or '\'')
            {
                var quote = c;
                var j = i + 1;

                while (j < code.Length && code[j] != quote)
                {
                    if (code[j] == '\\')
                    {
                        buffer[j] = ' ';
                        j++;
                    }

                    if (j < code.Length)
                    {
                        buffer[j] = ' ';
                        j++;
                    }
                }

                i = j;   // 閉じ引用符はそのまま残す (対応が読めるように)
                continue;
            }
        }

        return new string(buffer);
    }

    // 部分文字列の出現位置を全部返す。
    private static List<int> Occurrences(string text, string needle)
    {
        var list = new List<int>();

        for (var i = text.IndexOf(needle, StringComparison.Ordinal); i >= 0;
             i = text.IndexOf(needle, i + needle.Length, StringComparison.Ordinal))
        {
            list.Add(i);
        }

        return list;
    }

    // フィールドへの**代入**の位置。比較 (==/!=) は数えない。複合代入 (+= / ??=) は数える。
    private static List<int> Assignments(string code, string field)
        => Regex.Matches(code, @"(?<![A-Za-z0-9_])" + Regex.Escape(field) + @"\s*(\?\?|\+|-|\*|/|%|&|\||\^)?=(?!=)")
                .Select(m => m.Index)
                .ToList();

    // フィールドに代入されている式 (= の右辺 … ; まで) を全部返す。
    private static List<string> AssignedExpressions(string code, string field)
    {
        var list = new List<string>();

        foreach (var i in Assignments(code, field))
        {
            var eq = code.IndexOf('=', i);
            var semi = code.IndexOf(';', eq);
            if (eq < 0 || semi < 0)
            {
                continue;
            }

            list.Add(Normalize(code[(eq + 1)..semi]));
        }

        return list;
    }

    // signature で始まるメソッドの本体範囲 (本体の中身 = 波括弧の内側)。
    private static (int Start, int End)? MethodBodyRange(string code, string signature)
    {
        var i = code.IndexOf(signature, StringComparison.Ordinal);
        if (i < 0)
        {
            return null;
        }

        var open = code.IndexOf('{', i);
        if (open < 0)
        {
            return null;
        }

        var end = MatchingBrace(code, open);
        return end < 0 ? null : (open + 1, end);
    }

    // メソッド本体の中の try / catch / finally ブロックの範囲。
    private static List<(int Start, int End)> KeywordBlocks(string code, (int Start, int End) method, string keyword)
    {
        var list = new List<(int, int)>();
        var body = code[method.Start..method.End];

        foreach (Match m in Regex.Matches(body, @"(?<![A-Za-z0-9_])" + keyword + @"(?![A-Za-z0-9_])"))
        {
            var open = code.IndexOf('{', method.Start + m.Index);
            if (open < 0 || open >= method.End)
            {
                continue;
            }

            var end = MatchingBrace(code, open);
            if (end < 0)
            {
                continue;
            }

            list.Add((open + 1, end));
        }

        return list;
    }

    private static (int Start, int End)? KeywordBlock(string code, (int Start, int End) method, string keyword)
    {
        var all = KeywordBlocks(code, method, keyword);
        return all.Count == 1 ? all[0] : null;
    }
}
