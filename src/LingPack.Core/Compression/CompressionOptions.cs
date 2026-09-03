namespace LingPack.Core.Compression;

public sealed class CompressionOptions
{
    /// <summary>Fraction of words to keep (0-1). Mutually exclusive with <see cref="TargetTokens"/>.</summary>
    public double? Rate { get; init; }

    /// <summary>Absolute number of words to keep. Mutually exclusive with <see cref="Rate"/>.</summary>
    public int? TargetTokens { get; init; }

    /// <summary>Words that are always kept regardless of their preserve-probability score.</summary>
    public IReadOnlyCollection<string> ForceKeepTokens { get; init; } = [];

    /// <summary>Safety margin (in tokens) reserved below the model's max sequence length when chunking.</summary>
    public int ChunkSafetyMargin { get; init; } = 8;

    public void Validate()
    {
        if (Rate is null && TargetTokens is null)
        {
            throw new ArgumentException("Either Rate or TargetTokens must be specified.");
        }

        if (Rate is not null && TargetTokens is not null)
        {
            throw new ArgumentException("Rate and TargetTokens are mutually exclusive.");
        }

        if (Rate is < 0 or > 1)
        {
            throw new ArgumentException("Rate must be between 0 and 1.");
        }

        if (TargetTokens is < 0)
        {
            throw new ArgumentException("TargetTokens must be non-negative.");
        }
    }
}
