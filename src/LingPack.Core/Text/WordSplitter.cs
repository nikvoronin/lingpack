namespace LingPack.Core.Text;

/// <summary>
/// Splits text into whitespace-separated words while preserving the exact leading whitespace run
/// of each word, so <c>Join(Split(x)) == x</c> holds for the "keep everything" case.
/// </summary>
public static class WordSplitter
{
    public static IReadOnlyList<Word> Split(string text)
    {
        var words = new List<Word>();
        var i = 0;
        var n = text.Length;

        while (i < n)
        {
            var wsStart = i;
            while (i < n && char.IsWhiteSpace(text[i]))
            {
                i++;
            }

            var leadingWhitespace = text[wsStart..i];

            if (i >= n)
            {
                break;
            }

            var wordStart = i;
            while (i < n && !char.IsWhiteSpace(text[i]))
            {
                i++;
            }

            words.Add(new Word(text[wordStart..i], leadingWhitespace, wordStart, i - wordStart));
        }

        return words;
    }

    /// <summary>Reconstructs the original text from a full (unfiltered) word list.</summary>
    public static string Join(IReadOnlyList<Word> words)
    {
        var sb = new System.Text.StringBuilder();
        foreach (var word in words)
        {
            sb.Append(word.LeadingWhitespace).Append(word.Text);
        }

        return sb.ToString();
    }
}
