using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace LingPack.Core.Classification;

/// <summary>
/// Runs the exported LLMLingua-2 token-classification ONNX model. Input names/dtype are resolved
/// from the model's own metadata at load time rather than hardcoded, since exported models vary in
/// whether they declare a <c>token_type_ids</c> input.
/// </summary>
public sealed class OnnxTokenClassifier : ITokenClassifier, IDisposable
{
    private readonly InferenceSession _session;
    private readonly string _inputIdsName;
    private readonly string _attentionMaskName;
    private readonly string? _tokenTypeIdsName;
    private readonly string _outputName;
    private readonly bool _useInt64Inputs;

    public OnnxTokenClassifier(string onnxModelPath)
    {
        _session = new InferenceSession(onnxModelPath);

        var inputNames = _session.InputMetadata.Keys.ToHashSet();
        _inputIdsName = RequireInput(inputNames, "input_ids");
        _attentionMaskName = RequireInput(inputNames, "attention_mask");
        _tokenTypeIdsName = inputNames.Contains("token_type_ids") ? "token_type_ids" : null;

        _outputName = _session.OutputMetadata.ContainsKey("logits")
            ? "logits"
            : _session.OutputMetadata.Keys.First();

        _useInt64Inputs = _session.InputMetadata[_inputIdsName].ElementType == typeof(long);
    }

    private static string RequireInput(HashSet<string> names, string expected)
        => names.Contains(expected)
            ? expected
            : throw new InvalidOperationException(
                $"ONNX model is missing expected input '{expected}'. Found: {string.Join(", ", names)}");

    public float[] GetPreserveProbabilities(int[] inputIds, int[] attentionMask, int[]? tokenTypeIds)
    {
        var seqLen = inputIds.Length;

        var inputs = new List<NamedOnnxValue>
        {
            CreateInput(_inputIdsName, inputIds),
            CreateInput(_attentionMaskName, attentionMask),
        };

        if (_tokenTypeIdsName is not null)
        {
            inputs.Add(CreateInput(_tokenTypeIdsName, tokenTypeIds ?? new int[seqLen]));
        }

        using var results = _session.Run(inputs);
        var logits = results.First(r => r.Name == _outputName).AsTensor<float>();

        var numLabels = logits.Dimensions[2];
        var preserveProbability = new float[seqLen];

        for (var i = 0; i < seqLen; i++)
        {
            var max = float.NegativeInfinity;
            for (var label = 0; label < numLabels; label++)
            {
                max = Math.Max(max, logits[0, i, label]);
            }

            double sumExp = 0;
            var exp = new double[numLabels];
            for (var label = 0; label < numLabels; label++)
            {
                exp[label] = Math.Exp(logits[0, i, label] - max);
                sumExp += exp[label];
            }

            preserveProbability[i] = (float)(exp[1] / sumExp);
        }

        return preserveProbability;
    }

    private NamedOnnxValue CreateInput(string name, int[] values)
    {
        if (_useInt64Inputs)
        {
            var tensor = new DenseTensor<long>([1, values.Length]);
            for (var i = 0; i < values.Length; i++)
            {
                tensor[0, i] = values[i];
            }

            return NamedOnnxValue.CreateFromTensor(name, tensor);
        }
        else
        {
            var tensor = new DenseTensor<int>([1, values.Length]);
            for (var i = 0; i < values.Length; i++)
            {
                tensor[0, i] = values[i];
            }

            return NamedOnnxValue.CreateFromTensor(name, tensor);
        }
    }

    public void Dispose() => _session.Dispose();
}
