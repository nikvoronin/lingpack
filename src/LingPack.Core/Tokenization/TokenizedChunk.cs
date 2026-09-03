namespace LingPack.Core.Tokenization;

/// <summary>
/// The tokenized form of one chunk of words, ready to feed to a token classifier.
/// <see cref="TokenWordIndex"/> maps each token position back to its chunk-local word index;
/// special tokens (BOS/EOS/PAD) map to -1.
/// </summary>
public sealed record TokenizedChunk(
    int[] InputIds,
    int[] AttentionMask,
    int[]? TokenTypeIds,
    int[] TokenWordIndex,
    int WordCount);
