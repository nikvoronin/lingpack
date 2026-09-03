using System.Text;
using LingPack.Core.Chunking;
using LingPack.Core.Classification;
using LingPack.Core.Text;
using LingPack.Core.Tokenization;

namespace LingPack.Core.Compression;

/// <summary>
/// Orchestrates the LLMLingua-2 pipeline: split into words, chunk under the model's max sequence
/// length, classify each token's preserve-probability, aggregate to word-level scores, select the
/// words to keep, and reassemble the compressed prompt.
/// </summary>
public sealed class PromptCompressor(ITokenizerAdapter tokenizer, ITokenClassifier classifier)
{
    public CompressionResult Compress(string text, CompressionOptions options)
    {
        options.Validate();

        var words = WordSplitter.Split(text);
        if (words.Count == 0)
        {
            return new CompressionResult(text, 0, 0, 0, 0, []);
        }

        var chunkRanges = WordChunker.Split(words, tokenizer, tokenizer.MaxModelLength, options.ChunkSafetyMargin);

        var scores = new float[words.Count];
        var tokenCounts = new int[words.Count];

        foreach (var range in chunkRanges)
        {
            ScoreChunk(words, range, scores, tokenCounts);
        }

        var forceKeep = new HashSet<string>(options.ForceKeepTokens, StringComparer.Ordinal);
        var kept = SelectKept(words, scores, tokenCounts, forceKeep, options);

        var compressedText = Reassemble(words, kept);

        var wordScores = new WordScore[words.Count];
        int keptWordCount = 0, originalTokenCount = 0, compressedTokenCount = 0;

        for (var i = 0; i < words.Count; i++)
        {
            wordScores[i] = new WordScore(words[i].Text, scores[i], kept[i]);
            originalTokenCount += tokenCounts[i];

            if (kept[i])
            {
                keptWordCount++;
                compressedTokenCount += tokenCounts[i];
            }
        }

        return new CompressionResult(
            compressedText,
            words.Count,
            keptWordCount,
            originalTokenCount,
            compressedTokenCount,
            wordScores);
    }

    private void ScoreChunk(IReadOnlyList<Word> words, Range range, float[] scores, int[] tokenCounts)
    {
        var (offset, length) = range.GetOffsetAndLength(words.Count);
        var chunkWords = new string[length];
        for (var i = 0; i < length; i++)
        {
            chunkWords[i] = words[offset + i].Text;
        }

        var tokenized = tokenizer.Tokenize(chunkWords);
        var preserveProbabilities = classifier.GetPreserveProbabilities(
            tokenized.InputIds, tokenized.AttentionMask, tokenized.TokenTypeIds);

        var sumByWord = new double[length];
        var countByWord = new int[length];

        for (var t = 0; t < tokenized.TokenWordIndex.Length; t++)
        {
            var w = tokenized.TokenWordIndex[t];
            if (w < 0)
            {
                continue;
            }

            sumByWord[w] += preserveProbabilities[t];
            countByWord[w]++;
        }

        for (var w = 0; w < length; w++)
        {
            var globalIndex = offset + w;
            scores[globalIndex] = countByWord[w] > 0 ? (float)(sumByWord[w] / countByWord[w]) : 0f;
            tokenCounts[globalIndex] = countByWord[w];
        }
    }

    private static bool[] SelectKept(
        IReadOnlyList<Word> words, float[] scores, int[] tokenCounts, HashSet<string> forceKeep, CompressionOptions options)
    {
        var n = words.Count;
        var kept = new bool[n];

        // Force-kept words first (regardless of score), then the rest ranked by score descending;
        // ties broken by original index for deterministic output.
        var ranked = Enumerable.Range(0, n)
            .OrderByDescending(i => forceKeep.Contains(words[i].Text))
            .ThenByDescending(i => scores[i])
            .ThenBy(i => i)
            .ToArray();

        if (options.Rate is double rate)
        {
            var keepCount = Math.Clamp((int)Math.Round(n * rate), 0, n);
            var forceKeepCount = ranked.Count(i => forceKeep.Contains(words[i].Text));
            keepCount = Math.Max(keepCount, forceKeepCount);

            for (var k = 0; k < keepCount; k++)
            {
                kept[ranked[k]] = true;
            }
        }
        else
        {
            var target = options.TargetTokens!.Value;
            var cumulative = 0;

            foreach (var i in ranked)
            {
                var isForce = forceKeep.Contains(words[i].Text);
                if (isForce || cumulative + tokenCounts[i] <= target)
                {
                    kept[i] = true;
                    cumulative += tokenCounts[i];
                }
            }
        }

        return kept;
    }

    private static string Reassemble(IReadOnlyList<Word> words, bool[] kept)
    {
        var sb = new StringBuilder();
        var firstEmitted = true;
        var previousKeptIndex = -1;

        for (var i = 0; i < words.Count; i++)
        {
            if (!kept[i])
            {
                continue;
            }

            if (firstEmitted)
            {
                sb.Append(words[i].Text);
                firstEmitted = false;
            }
            else if (previousKeptIndex == i - 1)
            {
                sb.Append(words[i].LeadingWhitespace).Append(words[i].Text);
            }
            else
            {
                sb.Append(' ').Append(words[i].Text);
            }

            previousKeptIndex = i;
        }

        return sb.ToString();
    }
}
