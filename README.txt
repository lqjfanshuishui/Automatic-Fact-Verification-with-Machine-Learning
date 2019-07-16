1.Using the Doc_index.ipynb file to build indexes for all the document names in all wiki files.
2.Then using the built index to search top-3 scored documents for each claim by running the Search_doc.ipynb.
  Input: claims
  Output: candidate documents(a json file)
3.Run get_candidate_sents.ipynb to get all the sentences and sentence IDs in all searched documents.
  Input: candidate documents json file
  Output: candidate sentences json file
4.Evidence_selection.ipynb does the evidence selection.
  Input: candidate sentences json file
  Output: claims with its selected evidences
5.allen.py handles the labelling of the claims.
  Input: claims with its selected evidences
  Output: claims with its label and selected evidences
------------------------------------------------------------
6.get_words.ipynb generates the training set for the word2vec model from the train.json
7.train_Word2vec.ipynb trains the word2vec model and save it.
