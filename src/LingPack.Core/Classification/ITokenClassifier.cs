namespace LingPack.Core.Classification;

/// <summary>
/// Scores each token with the probability that it should be preserved (vs. discarded) by the
/// compressor. A pure array-in/array-out contract so it is trivially fakeable in tests without any
/// dependency on ONNX Runtime or a real downloaded model.
/// </summary>
public interface ITokenClassifier
{
    /// <summary>Returns one preserve-probability per input token, aligned 1:1 with <paramref name="inputIds"/>.</summary>
    float[] GetPreserveProbabilities(int[] inputIds, int[] attentionMask, int[]? tokenTypeIds);
}
