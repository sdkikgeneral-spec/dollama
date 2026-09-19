using Dollama.Ui.Services;
using Xunit;

namespace Dollama.Ui.Tests;

// EndpointUrl.Combine の 4 通りのスラッシュ有無を検証する。
public sealed class EndpointUrlTests
{
    [Theory]
    [InlineData("http://127.0.0.1:8080", "/health", "http://127.0.0.1:8080/health")]
    [InlineData("http://127.0.0.1:8080/", "/health", "http://127.0.0.1:8080/health")]
    [InlineData("http://127.0.0.1:8080", "health", "http://127.0.0.1:8080/health")]
    [InlineData("http://127.0.0.1:8080/", "health", "http://127.0.0.1:8080/health")]
    public void Combine_NormalizesSlashes(string baseUrl, string path, string expected)
    {
        Assert.Equal(expected, EndpointUrl.Combine(baseUrl, path));
    }
}
