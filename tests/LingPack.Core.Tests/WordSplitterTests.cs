using LingPack.Core.Text;

namespace LingPack.Core.Tests;

public class WordSplitterTests
{
    [Fact]
    public void Split_SingleSpaceSeparated_ProducesExpectedWords()
    {
        var words = WordSplitter.Split("hello world");

        Assert.Equal(2, words.Count);
        Assert.Equal("hello", words[0].Text);
        Assert.Equal("", words[0].LeadingWhitespace);
        Assert.Equal("world", words[1].Text);
        Assert.Equal(" ", words[1].LeadingWhitespace);
    }

    [Fact]
    public void Split_MultipleInternalSpaces_PreservesExactWhitespaceRun()
    {
        var words = WordSplitter.Split("a   b");

        Assert.Equal(2, words.Count);
        Assert.Equal("   ", words[1].LeadingWhitespace);
    }

    [Fact]
    public void Split_LeadingWhitespace_CapturedOnFirstWord()
    {
        var words = WordSplitter.Split("  hello");

        Assert.Single(words);
        Assert.Equal("  ", words[0].LeadingWhitespace);
        Assert.Equal("hello", words[0].Text);
    }

    [Fact]
    public void Split_NewlinesAndTabs_TreatedAsWhitespace()
    {
        var words = WordSplitter.Split("a\nb\tc");

        Assert.Equal(3, words.Count);
        Assert.Equal("\n", words[1].LeadingWhitespace);
        Assert.Equal("\t", words[2].LeadingWhitespace);
    }

    [Fact]
    public void Split_EmptyString_ProducesNoWords()
    {
        Assert.Empty(WordSplitter.Split(""));
    }

    [Fact]
    public void Split_WhitespaceOnlyString_ProducesNoWords()
    {
        Assert.Empty(WordSplitter.Split("   \n\t"));
    }

    [Theory]
    [InlineData("hello world")]
    [InlineData("  leading and   internal    spacing")]
    [InlineData("single")]
    [InlineData("line1\nline2\ttabbed")]
    public void Join_OfSplit_RoundTripsExactlyForTextWithoutTrailingWhitespace(string text)
    {
        var words = WordSplitter.Split(text);

        Assert.Equal(text, WordSplitter.Join(words));
    }

    [Fact]
    public void Split_TrailingWhitespace_IsNotPreservedInReassembly()
    {
        // Documented simplification: only whitespace preceding a word is captured, so trailing
        // whitespace at the very end of the input has nothing to attach to and is dropped.
        var words = WordSplitter.Split("hello world   ");

        Assert.Equal("hello world", WordSplitter.Join(words));
    }
}
