using Microsoft.ML.Tokenizers;

namespace LingPack.Core.Tokenization;

/// <summary>
/// Tokenizes words the way the exported <c>microsoft/llmlingua-2-xlm-roberta-large-meetingbank</c> ONNX model
/// expects: SentencePiece Unigram segmentation (via <see cref="SentencePieceTokenizer"/>) for piece
/// splitting, then id resolution via <see cref="HfVocab"/> — the authoritative piece-to-id mapping
/// from the exported <c>tokenizer.json</c>, not the raw ids the SentencePiece proto itself reports.
/// </summary>
public sealed class XlmRobertaTokenizerAdapter : ITokenizerAdapter
{
    private readonly SentencePieceTokenizer _sentencePieceTokenizer;
    private readonly HfVocab _vocab;

    public int MaxModelLength { get; }

    public XlmRobertaTokenizerAdapter(string sentencePieceModelPath, string tokenizerJsonPath, int maxModelLength = 512)
    {
        using (var modelStream = File.OpenRead(sentencePieceModelPath))
        {
            _sentencePieceTokenizer = SentencePieceTokenizer.Create(modelStream, addBeginningOfSentence: false, addEndOfSentence: false);
        }

        _vocab = HfVocab.LoadFromFile(tokenizerJsonPath);
        MaxModelLength = maxModelLength;
    }

    public int CountTokens(string word)
        => _sentencePieceTokenizer.CountTokens(word, addBeginningOfSentence: false, addEndOfSentence: false);

    public TokenizedChunk Tokenize(IReadOnlyList<string> chunkWords)
    {
        var inputIds = new List<int> { _vocab.BosId };
        var tokenWordIndex = new List<int> { -1 };

        for (var w = 0; w < chunkWords.Count; w++)
        {
            var pieces = _sentencePieceTokenizer.EncodeToTokens(
                chunkWords[w],
                out _,
                addBeginningOfSentence: false,
                addEndOfSentence: false);

            foreach (var piece in pieces)
            {
                inputIds.Add(_vocab.GetId(piece.Value));
                tokenWordIndex.Add(w);
            }
        }

        inputIds.Add(_vocab.EosId);
        tokenWordIndex.Add(-1);

        var attentionMask = new int[inputIds.Count];
        Array.Fill(attentionMask, 1);

        return new TokenizedChunk(
            InputIds: inputIds.ToArray(),
            AttentionMask: attentionMask,
            TokenTypeIds: null,
            TokenWordIndex: tokenWordIndex.ToArray(),
            WordCount: chunkWords.Count);
    }
}
