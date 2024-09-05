import numpy as np
import pickle
import torch
import sys
sys.path.append('..');
from module import (
    mutualRetrieval,
)
def reportRetrieval(
        score_per_A: torch.Tensor,
        score_per_B: torch.Tensor,
        AB_answers: torch.Tensor,
        BA_answers: torch.Tensor,
        metadata: dict = {
            "modality_A_title": "audio",
            "modality_B_title": "image",
            "modality_A_logAbbr": "A",
            "modality_B_logAbbr": "I",
        },
    ):
        """reportRetrieval

        Args:
            score_per_A (torch.Tensor): the similarity score per modality A sample
            score_per_B (torch.Tensor): the similarity score per modality B sample
            AB_answers (torch.Tensor): the golden answer (pair ID) for each audio sample
            BA_answers (torch.Tensor): the golden answer (pair ID) for each image sample
            metadata (dict): metadata should include modality the title for A, B and the abbreviation for A and B
        """

        # metadata should include modality the title for A, B and the abbreviation for A and B
        assert "modality_A_title" in metadata
        assert "modality_B_title" in metadata
        assert "modality_A_logAbbr" in metadata
        assert "modality_B_logAbbr" in metadata

        recall_results_AB, recall_results_BA, recall_results_mean = mutualRetrieval(
            score_per_A=score_per_A,
            score_per_B=score_per_B,
            AB_answers=AB_answers,
            BA_answers=BA_answers,
            recall_at=[1,5,10],
            modality_A_title=metadata["modality_A_title"],
            modality_B_title=metadata["modality_B_title"],
        )

        log_AB_abbr = "{}{}".format(
            metadata["modality_A_logAbbr"], metadata["modality_B_logAbbr"]
        )
        log_BA_abbr = "{}{}".format(
            metadata["modality_B_logAbbr"], metadata["modality_A_logAbbr"]
        )
        
    
        
        print(f"val_recall_{log_AB_abbr}", recall_results_AB)
        print(f"val_recall_{log_BA_abbr}", recall_results_BA)
        print("val_recall_mean", recall_results_mean)
        
        return recall_results_AB,recall_results_BA
def Bi_Retrieval(all_A_feats,all_B_feats,all_A_feats_id,all_B_feats_id,feat_A_name,feat_B_name ):
        # calculate dot product
        score_per_A = torch.matmul(
            all_A_feats.float(),
            all_B_feats.float().T
        )
        score_per_B = score_per_A.T

        # AI : Audio -> Image, IA: Image -> Audio
        AB_answers = all_A_feats_id
        BA_answers = all_B_feats_id

        recall_results_AB,recall_results_BA = reportRetrieval(
            score_per_A=score_per_A,
            score_per_B=score_per_B,
            AB_answers=AB_answers,
            BA_answers=BA_answers,
            metadata = {
            "modality_A_title": feat_A_name,
            "modality_B_title": feat_B_name,
            "modality_A_logAbbr": feat_A_name[0],
            "modality_B_logAbbr": feat_B_name[0],
        },
        )
        return  recall_results_AB,recall_results_BA

def retrieval(ids, images, texts):
    all_ids = torch.Tensor(ids)
    all_imgs =  torch.Tensor(images) #torch.cat([x["image_feat"] for x in outputs], dim=0)
    id_img_pairs = {_id.item(): _img for _id, _img in zip(all_ids, all_imgs)}

    del all_imgs

    all_audo_feats = torch.Tensor(texts) # torch.cat([x["audio_feat"] for x in outputs], dim=0)
    all_audo_feats_id = all_ids

    all_img_feats = torch.stack([x for _, x in id_img_pairs.items()], dim=0)
    all_img_feats_id = torch.Tensor(list(id_img_pairs.keys()))

    print(
        "Total #{} images, #{} audio".format(
            len(all_img_feats), len(all_audo_feats)
        )
    )

    # AI : Audio -> Image, IA: Image -> Audio
  
    recall_results_AB,recall_results_BA = Bi_Retrieval(all_img_feats,all_audo_feats,all_img_feats_id,all_audo_feats_id,'img','aud')
    recall_results_AB = list(recall_results_AB.values())
    recall_results_BA = list(recall_results_BA.values())
    
    outfile = f"& {recall_results_AB[0] :.1f} & {recall_results_AB[1] :.1f} & {recall_results_AB[2] :.1f} "
    outfile += f"& {recall_results_BA[0] :.1f} & {recall_results_BA[1] :.1f} & {recall_results_BA[2] :.1f} "
    print(outfile)

with open('locnarr_whisper_results_cache2.pkl', 'rb') as handle:
    results = pickle.load(handle)
id_list, image_feat_list, text_feat_list, whisper_texts, true_texts,\
          img_to_text_idx, text_to_img_idx, id_to_img_idx = results

retrieval(id_list,image_feat_list,text_feat_list)

from torchmetrics.functional import word_error_rate
whisper_lower = [text.lower() for text in whisper_texts]
true_lower = [text.lower() for text in true_texts]
WERs = [float(word_error_rate(whisper_lower[i], true_lower[i])) for i in range(len(whisper_lower))]
overall_WER = word_error_rate(whisper_lower, true_lower)
print(overall_WER)
from sklearn.neighbors import KDTree
def get_recall(source_feat_list, target_feat_list, target_idx_to_source_idx_dict, recall_levels=[1, 5, 10]):
    # source_feat_list is the list of embeddings in the query modality
    # target_feat_list is the list of embeddings in the target modality
    # target_idx_to_source_idx_dict is a dictionary going from an index in target_feat_list to a list of indeces in source_feat_list
    # used to determine if a given target is correct for a given source
    tree = KDTree(target_feat_list)
    idxs = np.array(tree.query(source_feat_list, k=recall_levels[-1])[1])
    a = idxs[0]
    closest_ids = [[target_idx_to_source_idx_dict[id] for id in idx] for idx in idxs]
    recall_avgs = []
    recalls = []
    for recall_level in recall_levels: 
        recall_at_level = []
        for source_idx in range(len(source_feat_list)):
            recalls_list = closest_ids[source_idx] # list of lists
            recalls_list = recalls_list[0:recall_level] # shortened to only contain ones for current recall level
            recalls_list_flat = []
            for a in recalls_list:
                recalls_list_flat += a
            recall_at_level += [source_idx in recalls_list_flat]
        print(f'recall at level {recall_level}: {np.mean(recall_at_level)}')
        recall_avgs += [np.mean(recall_at_level)]
        recalls += [recall_at_level]
    return recalls, recall_avgs

print('CLIP base, locnarr:')
print('text to image recall')
a = get_recall(text_feat_list, image_feat_list, text_to_img_idx)
print('image to text recall')
# a = get_recall(image_feat_list, text_feat_list, img_to_text_idx)


from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

recall_at_5 = a[0][1]
print(recall_at_5)
plt.scatter(WERs, recall_at_5)
x_new = np.linspace(0, 1, 50)
logreg_x = np.array(WERs)[:,np.newaxis]
logreg_y = np.array(recall_at_5).astype(int)
logreg = LogisticRegression()
logreg.fit(logreg_x, logreg_y)
y_new = logreg.predict_proba(x_new[:,None])
plt.plot(x_new, y_new[:,1])
plt.xlabel('Per-sample WER')
plt.ylabel('Per-sample recall')
plt.title('Speech to Image R@5 vs. per-sample WER, Whisper+CLIP, locnarr')
plt.savefig('sss.jpg')