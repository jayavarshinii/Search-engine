
# coding: utf-8

# In[30]:


import re
import os
import collections
import math
import timeit
import time

class index:
    
    def __init__(self,path):
        self.src_directory = path
        self.token_count_in_doc = {}
        (self.doc_tokens, self.doc_list) = self.buildIndex()
    
    def tokenize(self,fileName, file_index):
        file = open(fileName,encoding='utf-8')
        read_content = file.read()
        tokens = []
        tokens = re.sub("[^a-zA-Z]", " ", read_content)
        tokens = tokens.lower()
        tokens = re.sub("(?:^|(?<= ))(a|an|and|are|as|at|be|by|for|from|has|he|in|is|it|its|of|on|that|the|to|was|were|will|with)(?:(?= )|$)", "", tokens).split()
        self.token_count_in_doc[file_index] = len(tokens)
        return tokens

    def buildIndex(self):
        start_time = time.time()
        doc_list = {}
        doc_tokens = {}
        files = os.listdir(self.src_directory)
        for file_index, x in enumerate(files):
            fileName = self.src_directory + x
            doc_id = (fileName.split('/')[len(fileName.split('/')) - 1]).split('-')[1].split('.')[0]
            tokens = self.tokenize(fileName, file_index)
            doc_list[file_index] = doc_id
            for index, token in enumerate(tokens):
                done = False
                if token in doc_tokens:
                    for doc_and_indices in doc_tokens[token]:
                        if file_index == doc_and_indices[0]:
                            doc_and_indices[1].append(index)
                            done = True
                            break
                    if not done:
                        doc_tokens[token].append([file_index, [index]])
                else:
                    doc_tokens[token] = [[file_index, [index]]]
        for key, value in doc_tokens.items():
            list_of_tuples = []
            for each_entry in doc_tokens[key]:
                list_of_tuples.append(tuple(each_entry))
            doc_tokens[key] = list_of_tuples
        N = len(doc_list)
        doc_tokens_temp = {}
        doc_list_temp = {}

        for doc_token, value in doc_tokens.items():
            doc_tokens_temp[doc_token] = []


            count_docs = len(value)
            idf = math.log10(N / count_docs)

            doc_tokens_temp[doc_token].append(idf)

            for each_doc in value:
                freq_in_doc = len(each_doc[1])
                tf = 1 + (math.log10(freq_in_doc))
                weight = tf * idf        

                doc_tokens_temp[doc_token].append((each_doc[0],weight,each_doc[1]))
        doc_tokens = doc_tokens_temp
        end_time = time.time()
        print("Time taken to build index: ",end_time-start_time)
        return (doc_tokens, doc_list)

    def exact_query(self, query_terms, k=10):
        start_time = time.time()
#         query=[]
#         doclist=[]
#         qdict=[]
        doc_ids=[]
        for term in query_terms:
            value = self.doc_tokens.get(term)
            if value:
                for each in value[1:]:
                    doc_ids.append(each[0])
        cosines = self.cosine(query_terms, doc_ids)
        end_time = time.time()
        print("Time taken for exact_query: ",end_time-start_time)
        return cosines[:k]

    
    def create_champion_list(self,k):
        champion_list = {}
        for doc_token, value in self.doc_tokens.items():
            champion_list[doc_token] = []
            for each in value[1:]:
                champion_list[doc_token].append((each[0],each[1]))
        champion_list_tmp = {}
        for doc_token, value in champion_list.items():
            champion_list_tmp[doc_token] = sorted(value, key=lambda tup: tup[1])
        champion_list = champion_list_tmp
        champion_list_tmp = {}
        for doc_token, value in champion_list.items():
            count = 0
            champion_list_tmp[doc_token] = []
            for each in value:
                count+=1
                if count > k:
                    break
                champion_list_tmp[doc_token].append(each)
        champion_list = champion_list_tmp
        return champion_list


    def get_common_docs(self,query_terms):
        terms_docs = {}
        champion_list_query_terms = query_terms
        for champion_list_query_term in champion_list_query_terms:
            champion_docids_tfidfs = self.champion_list.get(champion_list_query_term)
            if champion_docids_tfidfs:
                terms_docs[champion_list_query_term] = []
                for each in champion_docids_tfidfs:
                    terms_docs[champion_list_query_term].append(each[0])
        common_docs = terms_docs[list(terms_docs.keys())[0]]
        lists = []
        for key in list(terms_docs.keys())[1:]:
            lists.append(terms_docs[key])
        for each in lists:
            common_docs = set(each)&set(common_docs)
        return common_docs

    def get_common_doc_ids(self,query_terms):
        terms_docs = {}
        for query_term in query_terms:
            terms_docs[query_term] = []
            value = self.doc_tokens.get(query_term)
            if value:
                for each in value[1:]:
                    terms_docs[query_term].append(each[0])
        if len(terms_docs)>0:
            common_docs = terms_docs[list(terms_docs.keys())[0]]
            lists = []
            for key in list(terms_docs.keys())[1:]:
                lists.append(terms_docs[key])
            for each in lists:
                common_docs = set(each)&set(common_docs)
        return common_docs
    
    def get_tfidf(self,query_term, doc_id):
        tfidfs = []
        value = self.doc_tokens.get(query_term)
        if value:
            for each in value[1:]:
                if each[0] == doc_id:
                    tfidfs.append(each[1])
                    break
            if len(tfidfs)>0:
                return tfidfs[0]
            else:
                return 0
        else:
            return 0

    def get_idf(self,query_term):
        if self.doc_tokens.get(query_term):
            return self.doc_tokens[query_term][0]
        else:
            return 0
    
    def cosine_for_each_doc(self,query_terms, doc_id):
        cosine = 0.0
        for query_term in query_terms:
            idf = self.get_idf(query_term)
            cosine = cosine + ((self.get_tfidf(query_term,doc_id)*idf)/self.token_count_in_doc[doc_id])
        return cosine

    def cosine(self,query_terms, doc_ids):
        cosines = []
        for doc_id in doc_ids:
            
            cosines.append([doc_id,self.cosine_for_each_doc(query_terms, doc_id)])
        return cosines

    
    def inexact_query_champion(self, query_terms, k=5):
        start_time = time.time()
    #function for exact top K retrieval using champion list (method 2)
    #Returns at the minimum the document names of the top K documents ordered in decreasing order of similarity score
        self.champion_list = self.create_champion_list(k)

        champion_list_query_terms = query_terms
        
        common_docs = self.get_common_doc_ids(query_terms)
        
        champion_list_cosines = sorted(self.cosine(champion_list_query_terms,common_docs),key=lambda tup: tup[1], reverse=True)[:k]
        end_time = time.time()
        print("Time taken for inexact_query_champion: ",end_time-start_time)
        return champion_list_cosines

    def inexact_query_index_elimination(self, query_terms, k=5):
    #function for exact top K retrieval using index elimination (method 3)
    #Returns at the minimum the document names of the top K documents ordered in decreasing order of similarity score
        start_time = time.time()
        index_elimination_query_terms = query_terms
        index_elimination_idfs = []
        for index_elimination_query_term in index_elimination_query_terms:
            index_elimination_idfs.append([index_elimination_query_term, self.get_idf(index_elimination_query_term)])
        sorted_index_elimination_idfs = sorted(index_elimination_idfs,key=lambda a: a[1], reverse=True)
        
        sorted_index_elimination_idfs = sorted_index_elimination_idfs[int(len(sorted_index_elimination_idfs)/2):]
        terms_sorted_index_elimination_idf = []
        for sorted_index_elimination_idf in sorted_index_elimination_idfs:
            terms_sorted_index_elimination_idf.append(sorted_index_elimination_idf[0])
        index_elimination_cosines = self.cosine(terms_sorted_index_elimination_idf, self.get_common_doc_ids(terms_sorted_index_elimination_idf))
        index_elimination_cosines=sorted(index_elimination_cosines, key=lambda tup: tup[1], reverse=True)[0]
        end_time = time.time()
        print("Time taken for inexact_query_index_elimination: ",end_time-start_time)
        return index_elimination_cosines[:k]
        

    def print_dict(self):
        #function to print the terms and posting list in the index
        print(self.doc_tokens)

    def print_doc_list(self):
    # function to print the documents and their document id
        print(self.doc_list)
        
def main():
    index_doc_retrieval = index('collection/')
    query_input = input("What are the query terms? Please provide input as comma separated values. E.g., raw,had,saw\n")
    query = query_input.split(',')
    print(index_doc_retrieval.exact_query(query))
    print(index_doc_retrieval.inexact_query_champion(query))
    print(index_doc_retrieval.inexact_query_index_elimination(query))

if __name__ == '__main__':
    main()


# In[73]:


from IPython.core.display import display, HTML
display(HTML("<style>.container { width:100% !important; }</style>"))


# In[3]:


a = {'a':'b'}


# In[5]:


len(a)

