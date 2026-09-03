using LingPack.Core.Tokenization;

namespace LingPack.Core.Tests.Fakes;

/// <summary>
/// A deterministic tokenizer for pure-logic tests: tokenizes each word into a configurable number
/// of tokens (default 1) without touching any real model or SentencePiece vocabulary.
/// </summary>
public sealed class FakeTokenizerAdapter(int maxModelLength = 512, Func<string, int>? tokenCountPerWord = null) : ITokenizerAdapter
{
    public int MaxModelLength { get; } = maxModelLength;

    public int CountTokens(string word) => tokenCountPerWord?.Invoke(word) ?? 1;

    public TokenizedChunk Tokenize(IReadOnlyList<string> chunkWords)
    {
        var ids = new List<int> { 0 }; // BOS
        var wordIndex = new List<int> { -1 };

        for (var w = 0; w < chunkWords.Count; w++)
        {
            var count = CountTokens(chunkWords[w]);
            for (var s = 0; s < count; s++)
            {
                ids.Add(ids.Count);
                wordIndex.Add(w);
            }
        }

        ids.Add(-2); // EOS
        wordIndex.Add(-1);

        var attentionMask = new int[ids.Count];
        Array.Fill(attentionMask, 1);

        return new TokenizedChunk(ids.ToArray(), attentionMask, null, wordIndex.ToArray(), chunkWords.Count);
    }
}
