using LingPack.Core.Chunking;
using LingPack.Core.Text;
using LingPack.Core.Tests.Fakes;

namespace LingPack.Core.Tests;

public class WordChunkerTests
{
    private static IReadOnlyList<Word> MakeWords(int count)
        => Enumerable.Range(0, count).Select(i => new Word($"w{i}", i == 0 ? "" : " ", i, 2)).ToList();

    [Fact]
    public void Split_WordsUnderBudget_ProducesSingleChunkCoveringAllWords()
    {
        var words = MakeWords(10);
        var tokenizer = new FakeTokenizerAdapter();

        var ranges = WordChunker.Split(words, tokenizer, maxModelLength: 512, safetyMargin: 8);

        var range = Assert.Single(ranges);
        var (offset, length) = range.GetOffsetAndLength(words.Count);
        Assert.Equal(0, offset);
        Assert.Equal(10, length);
    }

    [Fact]
    public void Split_LongInput_SplitsIntoMultipleChunksThatCoverAllWordsContiguously()
    {
        var words = MakeWords(100);
        var tokenizer = new FakeTokenizerAdapter(); // 1 token/word

        // budget = maxModelLength - 2 (BOS/EOS) - safetyMargin(0) = 8 tokens/words per chunk
        var ranges = WordChunker.Split(words, tokenizer, maxModelLength: 10, safetyMargin: 0);

        Assert.True(ranges.Count > 1);

        var covered = new bool[words.Count];
        var previousEnd = 0;
        foreach (var range in ranges)
        {
            var (offset, length) = range.GetOffsetAndLength(words.Count);
            Assert.Equal(previousEnd, offset); // contiguous, no gaps/overlaps
            for (var i = offset; i < offset + length; i++)
            {
                Assert.False(covered[i]);
                covered[i] = true;
            }
            previousEnd = offset + length;
        }

        Assert.All(covered, Assert.True);
        Assert.Equal(words.Count, previousEnd);
    }

    [Fact]
    public void Split_ChunkTokenBudget_NeverExceedsBudgetExceptForASingleOversizedWord()
    {
        var words = MakeWords(50);
        var tokenizer = new FakeTokenizerAdapter();
        const int maxModelLength = 20;
        const int safetyMargin = 2;
        var budget = maxModelLength - 2 - safetyMargin;

        var ranges = WordChunker.Split(words, tokenizer, maxModelLength, safetyMargin);

        foreach (var range in ranges)
        {
            var (offset, length) = range.GetOffsetAndLength(words.Count);
            var tokenCount = Enumerable.Range(offset, length).Sum(i => tokenizer.CountTokens(words[i].Text));
            Assert.True(length == 1 || tokenCount <= budget);
        }
    }

    [Fact]
    public void Split_SingleWordLargerThanBudget_DoesNotHangAndFormsItsOwnChunk()
    {
        var words = MakeWords(3);
        var tokenizer = new FakeTokenizerAdapter(tokenCountPerWord: _ => 1000); // every word blows the budget

        var ranges = WordChunker.Split(words, tokenizer, maxModelLength: 10, safetyMargin: 0);

        Assert.Equal(3, ranges.Count);
        foreach (var range in ranges)
        {
            Assert.Equal(1, range.GetOffsetAndLength(words.Count).Length);
        }
    }

    [Fact]
    public void Split_EmptyWordList_ProducesNoChunks()
    {
        var ranges = WordChunker.Split([], new FakeTokenizerAdapter(), maxModelLength: 512);

        Assert.Empty(ranges);
    }
}
