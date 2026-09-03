namespace LingPack.Core.Compression;

public sealed record CompressionResult(
    string CompressedText,
    int OriginalWordCount,
    int KeptWordCount,
    int OriginalTokenCount,
    int CompressedTokenCount,
    IReadOnlyList<WordScore> WordScores)
{
    public double AchievedRate => OriginalWordCount == 0 ? 1.0 : (double)KeptWordCount / OriginalWordCount;
}
