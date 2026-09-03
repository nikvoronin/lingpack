using System.CommandLine;
using LingPack.Cli;

var rootCommand = new RootCommand("lingpack — LLMLingua-2 prompt compressor")
{
    CompressCommandBuilder.Build(),
};

var parseResult = rootCommand.Parse(args);
return await parseResult.InvokeAsync();
