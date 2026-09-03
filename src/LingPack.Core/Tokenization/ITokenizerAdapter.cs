namespace LingPack.Core.Tokenization;

public interface ITokenizerAdapter
{
    /// <summary>Maximum number of tokens (including BOS/EOS) the underlying model accepts.</summary>
    int MaxModelLength { get; }

    /// <summary>Number of subword tokens a single word tokenizes to, ignoring BOS/EOS. Used for chunk sizing.</summary>
    int CountTokens(string word);

    /// <summary>Tokenizes a contiguous run of words (a chunk) into model-ready input arrays.</summary>
    TokenizedChunk Tokenize(IReadOnlyList<string> chunkWords);
}
