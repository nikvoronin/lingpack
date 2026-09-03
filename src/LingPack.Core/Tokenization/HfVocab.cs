using System.Text.Json;

namespace LingPack.Core.Tokenization;

/// <summary>
/// The authoritative piece-to-id vocabulary for the exported model, sourced directly from the
/// Hugging Face "fast tokenizer" <c>tokenizer.json</c>. This is the ground truth for whatever ids
/// the model was actually fine-tuned/exported against — we deliberately avoid re-deriving ids from
/// the raw SentencePiece proto's own internal numbering (see README "tokenizer id mapping" note).
/// </summary>
public sealed class HfVocab
{
    private readonly Dictionary<string, int> _pieceToId;

    public int UnkId { get; }
    public int BosId { get; }
    public int EosId { get; }
    public int PadId { get; }
    public int? MaskId { get; }

    private HfVocab(Dictionary<string, int> pieceToId, int unkId, int bosId, int eosId, int padId, int? maskId)
    {
        _pieceToId = pieceToId;
        UnkId = unkId;
        BosId = bosId;
        EosId = eosId;
        PadId = padId;
        MaskId = maskId;
    }

    public static HfVocab LoadFromFile(string tokenizerJsonPath)
    {
        using var stream = File.OpenRead(tokenizerJsonPath);
        using var doc = JsonDocument.Parse(stream);
        var root = doc.RootElement;

        var pieceToId = new Dictionary<string, int>();

        // model.vocab is a Unigram vocabulary: an ordered array of [piece, score] pairs whose
        // array index IS the token id used at inference time by this exported model.
        if (root.TryGetProperty("model", out var model) && model.TryGetProperty("vocab", out var vocab))
        {
            var index = 0;
            foreach (var entry in vocab.EnumerateArray())
            {
                var piece = entry[0].GetString() ?? throw new FormatException("tokenizer.json vocab entry missing piece string.");
                pieceToId[piece] = index;
                index++;
            }
        }

        // added_tokens carries the authoritative ids for specials (and any tokens appended after
        // the base vocab, e.g. <mask>); it overrides whatever model.vocab said for the same piece.
        string? bos = null, eos = null, pad = null, unk = null, mask = null;
        var specialIds = new Dictionary<string, int>();

        if (root.TryGetProperty("added_tokens", out var addedTokens))
        {
            foreach (var token in addedTokens.EnumerateArray())
            {
                var content = token.GetProperty("content").GetString()
                    ?? throw new FormatException("tokenizer.json added_tokens entry missing content.");
                var id = token.GetProperty("id").GetInt32();
                pieceToId[content] = id;
                specialIds[content] = id;
            }
        }

        bos = FindSpecial(specialIds, "<s>");
        eos = FindSpecial(specialIds, "</s>");
        pad = FindSpecial(specialIds, "<pad>");
        unk = FindSpecial(specialIds, "<unk>");
        mask = FindSpecial(specialIds, "<mask>");

        if (bos is null || eos is null || pad is null || unk is null)
        {
            throw new FormatException(
                $"tokenizer.json at '{tokenizerJsonPath}' is missing one or more required special tokens (<s>, </s>, <pad>, <unk>) in added_tokens.");
        }

        return new HfVocab(
            pieceToId,
            unkId: specialIds[unk],
            bosId: specialIds[bos],
            eosId: specialIds[eos],
            padId: specialIds[pad],
            maskId: mask is not null ? specialIds[mask] : null);
    }

    private static string? FindSpecial(Dictionary<string, int> specialIds, string content)
        => specialIds.ContainsKey(content) ? content : null;

    public int GetId(string piece) => _pieceToId.TryGetValue(piece, out var id) ? id : UnkId;
}
