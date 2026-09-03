using LingPack.Core.Classification;

namespace LingPack.Core.Tests.Fakes;

/// <summary>
/// Returns pre-scripted preserve-probability arrays, one per call, in order — one array per chunk
/// the test expects <see cref="LingPack.Core.Compression.PromptCompressor"/> to classify.
/// </summary>
public sealed class FakeTokenClassifier(params float[][] responses) : ITokenClassifier
{
    private readonly Queue<float[]> _responses = new(responses);

    public List<int[]> Calls { get; } = [];

    public float[] GetPreserveProbabilities(int[] inputIds, int[] attentionMask, int[]? tokenTypeIds)
    {
        Calls.Add(inputIds);

        if (_responses.Count == 0)
        {
            throw new InvalidOperationException("FakeTokenClassifier ran out of scripted responses.");
        }

        var response = _responses.Dequeue();

        if (response.Length != inputIds.Length)
        {
            throw new InvalidOperationException(
                $"Scripted response length {response.Length} does not match input length {inputIds.Length}.");
        }

        return response;
    }
}
