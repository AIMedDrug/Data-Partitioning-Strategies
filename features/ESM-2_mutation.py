import os
import pathlib
import torch
from esm import FastaBatchedDataset, pretrained

def extract_embeddings(model_name, fasta_file, output_dir, tokens_per_batch=4096, seq_length=1022, repr_layers=[33]):
    model, alphabet = pretrained.load_model_and_alphabet(model_name)
    model.eval()

    if torch.cuda.is_available():
        model = model.cuda()
        
    dataset = FastaBatchedDataset.from_file(fasta_file)
    batches = dataset.get_batch_indices(tokens_per_batch, extra_toks_per_seq=1)

    data_loader = torch.utils.data.DataLoader(
        dataset, 
        collate_fn=alphabet.get_batch_converter(seq_length), 
        batch_sampler=batches
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    
    with torch.no_grad():
        for batch_idx, (labels, strs, toks) in enumerate(data_loader):

            print(f'Processing batch {batch_idx + 1} of {len(batches)}')

            if torch.cuda.is_available():
                toks = toks.to(device="cuda", non_blocking=True)

            out = model(toks, repr_layers=repr_layers, return_contacts=False)

            logits = out["logits"].to(device="cpu")
            representations = {layer: t.to(device="cpu") for layer, t in out["representations"].items()}
            
            for i, label in enumerate(labels):
                entry_id = label.split()[0]
                
                filename = output_dir / f"{entry_id}.pt"
                truncate_len = min(seq_length, len(strs[i]))

                result = {"entry_id": entry_id}
                result["mean_representations"] = {
                        layer: t[i, 1 : truncate_len + 1].mean(0).clone()
                        for layer, t in representations.items()
                    }

                torch.save(result, filename)

# 设置模型名称
model_name = 'esm2_t33_650M_UR50D'

# 定义输入文件夹和输出文件夹路径
input_dir = pathlib.Path('/home/data1/BGM/fenziduiqi/fasta突变文件')
train_output_dir = pathlib.Path('mutation_train_embeddings')
test_output_dir = pathlib.Path('mutation_test_embeddings')

# 获取输入文件夹下的所有 .fasta 文件
fasta_files = list(input_dir.glob('*.fasta'))

# 处理所有 .fasta 文件
for fasta_file in fasta_files:
    print(f"Processing {fasta_file}")
    # 生成训练集嵌入
    #extract_embeddings(model_name, fasta_file, train_output_dir)
    
    # 生成测试集嵌入
    extract_embeddings(model_name, fasta_file, test_output_dir)

print("嵌入提取完成。")
