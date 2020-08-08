from ner.utils import load_model, extend_maps, prepocess_data_for_lstmcrf
from ner.data import build_corpus
from ner.evaluating import Metrics
from ner.evaluate import ensemble_evaluate

CRF_MODEL_PATH = 'data/model/crf.pkl'
BiLSTM_MODEL_PATH = 'data/model/bilstm.pkl'
BiLSTMCRF_MODEL_PATH = 'data/model/bilstm_crf.pkl'

REMOVE_O = False  # 在评估的时候是否去除O标记


def main():
    print("读取数据...")
    train_word_lists, train_tag_lists, word2id, tag2id = \
        build_corpus("train")
    dev_word_lists, dev_tag_lists = build_corpus("dev", make_vocab=False)
    test_word_lists, test_tag_lists = build_corpus("test", make_vocab=False)

    # 加载并评估CRF模型
    print("加载并评估crf模型...")
    crf_model = load_model(CRF_MODEL_PATH)
    crf_pred = crf_model.test(test_word_lists)
    metrics = Metrics(test_tag_lists, crf_pred, remove_O=REMOVE_O)
    metrics.report_scores()
    metrics.report_confusion_matrix()


    print("加载并评估bilstm+crf模型...")
    crf_word2id, crf_tag2id = extend_maps(word2id, tag2id, for_crf=True)
    bilstm_model = load_model(BiLSTMCRF_MODEL_PATH)
    bilstm_model.model.bilstm.bilstm.flatten_parameters()  # remove warning
    test_word_lists, test_tag_lists = prepocess_data_for_lstmcrf(
        test_word_lists, test_tag_lists, test=True
    )
    lstmcrf_pred, target_tag_list = bilstm_model.test(test_word_lists, test_tag_lists,
                                                      crf_word2id, crf_tag2id)
    metrics = Metrics(target_tag_list, lstmcrf_pred, remove_O=REMOVE_O)
    metrics.report_scores()
    metrics.report_confusion_matrix()

    ensemble_evaluate(
        [crf_pred, lstmcrf_pred],
        test_tag_lists
    )


if __name__ == "__main__":
    main()