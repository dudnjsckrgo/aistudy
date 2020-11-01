from GANAttentionTextGenerator import Lang , GeneratorTransformerModel,DiscriminatorTransformerModel,PositionalEncoding \
    ,readLangs,weights_init
filename="problemcorpus"
# import pandas as pd
# df = pd.read_csv(filename)
# print(df)
# mylist = []
# df['corpus'].apply(lambda x :mylist.append(x.lower()) )
# print(mylist)
# text="\n".join(mylist)
# # print(text)

# with open("problemcorpus.txt","wt") as f:
#     f.write(text)
# with open("problemcorpus.txt","rt") as f:
#     lines=f.readlines()
#     for line in lines:
#         print(line)


input_lang, pairs=readLangs(filename)
print("input_lang=================================\n",input_lang)
# print("pairs======================\n",pairs)