from langchain_community.retrievers import BM25Retriever
from langchain_core.runnables import RunnableLambda  # 引入这个来包装我们的手写算法
from langchain_chroma import Chroma
from langchain_core.documents import Document
from utils.config_handler import chroma_conf
from model.factory import embed_model
from langchain_text_splitters import RecursiveCharacterTextSplitter
from utils.path_tool import get_abs_path
from utils.file_handler import pdf_loader, txt_loader, listdir_with_allowed_type, get_file_md5_hex
from utils.logger_handler import logger
import os


class VectorStoreService:
    def __init__(self):
        self.vector_store = Chroma(
            collection_name=chroma_conf["collection_name"],
            embedding_function=embed_model,
            persist_directory=chroma_conf["persist_directory"],
        )

        self.spliter = RecursiveCharacterTextSplitter(
            chunk_size=chroma_conf["chunk_size"],
            chunk_overlap=chroma_conf["chunk_overlap"],
            separators=chroma_conf["separators"],
            length_function=len,
        )

    def get_retriever(self):
        # 1. 基础的向量检索器 (稠密检索)
        vector_retriever = self.vector_store.as_retriever(search_kwargs={"k": chroma_conf["k"]})

        # 2. 提取数据库中已有的文本，用于构建 BM25
        db_data = self.vector_store.get()
        all_texts = db_data.get('documents', [])

        if not all_texts:
            logger.warning("知识库文本为空，将回退到基础向量检索。请先运行 load_document()")
            return vector_retriever

        # 3. 初始化 BM25 检索器 (稀疏关键字检索)
        bm25_retriever = BM25Retriever.from_texts(all_texts)
        bm25_retriever.k = chroma_conf["k"]

        # ==================== 核心亮点：手写 RRF 混合检索算法 ====================
        def custom_ensemble_retriever(query: str) -> list[Document]:
            # 分别获取两路召回的结果
            vec_docs = vector_retriever.invoke(query)
            bm25_docs = bm25_retriever.invoke(query)

            # RRF (Reciprocal Rank Fusion) 倒数排名融合逻辑
            rrf_scores = {}
            c = 60  # RRF 算法的平滑常数

            # 处理向量检索结果打分
            for rank, doc in enumerate(vec_docs):
                if doc.page_content not in rrf_scores:
                    rrf_scores[doc.page_content] = {'score': 0.0, 'doc': doc}
                rrf_scores[doc.page_content]['score'] += 0.5 / (rank + 1 + c)

            # 处理 BM25 检索结果打分
            for rank, doc in enumerate(bm25_docs):
                if doc.page_content not in rrf_scores:
                    rrf_scores[doc.page_content] = {'score': 0.0, 'doc': doc}
                rrf_scores[doc.page_content]['score'] += 0.5 / (rank + 1 + c)

            # 按最终融合得分降序排序，并截取 Top K
            sorted_items = sorted(rrf_scores.values(), key=lambda x: x['score'], reverse=True)
            return [item['doc'] for item in sorted_items[:chroma_conf["k"]]]

        # =========================================================================

        # 用 RunnableLambda 将 Python 函数包装成 LangChain 标准组件
        return RunnableLambda(custom_ensemble_retriever)

    def load_document(self):
        """
        从数据文件夹内读取数据文件，转为向量存入向量库
        要计算文件的MD5做去重
        :return: None
        """

        def check_md5_hex(md5_for_check: str):
            if not os.path.exists(get_abs_path(chroma_conf["md5_hex_store"])):
                open(get_abs_path(chroma_conf["md5_hex_store"]), "w", encoding="utf-8").close()
                return False

            with open(get_abs_path(chroma_conf["md5_hex_store"]), "r", encoding="utf-8") as f:
                for line in f.readlines():
                    line = line.strip()
                    if line == md5_for_check:
                        return True
                return False

        def save_md5_hex(md5_for_check: str):
            with open(get_abs_path(chroma_conf["md5_hex_store"]), "a", encoding="utf-8") as f:
                f.write(md5_for_check + "\n")

        def get_file_documents(read_path: str):
            if read_path.endswith("txt"):
                return txt_loader(read_path)
            if read_path.endswith("pdf"):
                return pdf_loader(read_path)
            return []

        allowed_files_path: list[str] = listdir_with_allowed_type(
            get_abs_path(chroma_conf["data_path"]),
            tuple(chroma_conf["allow_knowledge_file_type"]),
        )

        for path in allowed_files_path:
            md5_hex = get_file_md5_hex(path)
            if check_md5_hex(md5_hex):
                logger.info(f"[加载知识库]{path}内容已经存在知识库内，跳过")
                continue

            try:
                documents: list[Document] = get_file_documents(path)
                if not documents:
                    continue

                split_document: list[Document] = self.spliter.split_documents(documents)
                if not split_document:
                    continue

                self.vector_store.add_documents(split_document)
                save_md5_hex(md5_hex)
                logger.info(f"[加载知识库]{path} 内容加载成功")
            except Exception as e:
                logger.error(f"[加载知识库]{path}加载失败：{str(e)}", exc_info=True)
                continue


if __name__ == '__main__':
    vs = VectorStoreService()
    vs.load_document()

    retriever = vs.get_retriever()

    # 这里我们测试一下你的专属领域数据检索效果
    test_query = "电价预测模型"  # 你可以把这里的词换成你放入文档里的核心名词
    print(f"\n======== 正在使用自研 RRF 混合检索算法搜索: [{test_query}] ========")

    res = retriever.invoke(test_query)
    for i, r in enumerate(res):
        print(f"\n--- 混合召回片段 {i + 1} ---")
        print(r.page_content)