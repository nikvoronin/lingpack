using LingPack.Core.Text;
using LingPack.Core.Tokenization;

namespace LingPack.Core.Chunking;

/// <summary>
/// Splits a word list into contiguous ranges whose tokenized length stays within the model's
/// max sequence length, using per-word token counts (cheap, and correct for SentencePiece Unigram
/// segmentation, which only merges within a word's own <c>▁</c>-prefixed boundary).
/// </summary>
public static class WordChunker
{
    public static IReadOnlyList<Range> Split(IReadOnlyList<Word> words, ITokenizerAdapter tokenizer, int maxModelLength, int safetyMargin = 8)
    {
        var budget = Math.Max(1, maxModelLength - 2 - safetyMargin); // reserve BOS/EOS + margin

        var ranges = new List<Range>();
        var chunkStart = 0;
        var runningTokens = 0;

        for (var i = 0; i < words.Count; i++)
        {
            var tokenCount = tokenizer.CountTokens(words[i].Text);

            if (i > chunkStart && runningTokens + tokenCount > budget)
            {
                ranges.Add(chunkStart..i);
                chunkStart = i;
                runningTokens = 0;
            }

            runningTokens += tokenCount;
        }

        if (chunkStart < words.Count)
        {
            ranges.Add(chunkStart..words.Count);
        }

        return ranges;
    }
}
