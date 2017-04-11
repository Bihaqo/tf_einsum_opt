# Optimizing TF's einsum function
 Einsum is a very powerful function for contracting tensors of arbitrary dimension and index.
 However, on TensorFlow it is implemented as a sequence of matrix-by-matrix multiplication in the order of arguments, so if you provide the arguments in a non-optimal order, einsum maybe very slow.
 
 For example usage see ```example.ipynb```.