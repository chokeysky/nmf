#!/usr/bin/python2.7
#
# COMP41450 Assigment 2
# NMF implementation with sparse term-document matrix
# Erik Nyquist 13206065

import numpy
import sys
import os

TERMS="bbcnews.terms"
TDMATRIX="bbcnews.mtx"
MINERROR=0.02
MAXITER=100

# VT100 terminal control codes
# for clearing the last console line
CLRLINE='\x1b[1A\x1b[2K'

def usage(arg0):
	print("\nUsage: " + arg0 + " K\nK = no. of clusters\n")
	sys.exit(1)

def malformed(msg):
	'''prints message & exits in the case of a malformed
	term-document matrix'''

	print("Malformed term-document matrix in file '" + TDMATRIX + "' :\n"
		+ msg)
	sys.exit(-1)

def populate_matrix():
	'''reads in a list of terms & a term-document matrix
	from supplied files, populates an array.
	returns: A, DF, terms
	A = term-document matrix in the form of 2D array of floats.
	DF = integer array, entry DF[i] quantifies no. of documents term i appears in
	terms = the terms as a 1D array of strings.'''

	files = ""
	if not os.path.exists(TERMS):
		files += (TERMS + "\n")
	if not os.path.exists(TDMATRIX):
		files += (TDMATRIX + "\n")

	if files != "":
		print("\nno such file(s) :\n" + files)
		sys.exit(-1)
	
	with open(TERMS, "r") as tfh:
		print("Reading file '" + TERMS + "'...")
		terms =	tfh.readlines()

	with open(TDMATRIX, "r") as mfh:
		print("Reading file '" + TDMATRIX + "' ...")
		header = mfh.readline() # skip header

		# read matrix dimensions
		rows, columns, numvalues = mfh.readline().split()

		# initialise term frequency (TF) matrix A with above dimensions
		A = [[0.0 for i in range(int(columns))] for j in range(int(rows))]

		# initialise array to store document frequency (DF) of terms
		DF = [0 for i in range(len(terms))]

		# populate matrix A
		count = 0
		for line in mfh:
			# skip empty lines- just in case!
			if (line.strip() == ''): 
				continue

			term, doc, freq = line.split()
			term = int(term)
			doc = int(doc)
			freq = float(freq)

			# sanity check:
			# double entry means I did something wrong,
			# or matrix was generated incorrectly
			if A[term -1][doc - 1] == 0:
				A[term - 1][doc - 1] = freq
				count += 1
			else:
				malformed("Entry #" + str(count + 1) + ", for term '" +
					terms[term - 1].strip() + "' in document #" + str(doc) +
					"\nalready has a value of " + A[term -1][doc -1] +
					".\nEntries cannot be assigned twice.")

			DF[term - 1] += 1

		# final sanity check
		if int(numvalues) != count:
				malformed("Expecting " + numvalues +
					" entries but found " + str(count))
	return A, DF, terms	

def tf_idf(A, DF):
	'''takes matrix A and performs TF-IDF normalisation'''
	n = len(A)
        print()
	for x in range(len(A)):
		for y in range(len(A[x])):
			if (A[x][y] != 0):
				A[x][y] *= numpy.log10(n / DF[y])
        	print(CLRLINE + "Normalising input matrix : %.2f%%"
			% (((x + 1) / float(n) * 100)))
            
def distance(A, B):
	'''returns the euclidian distance between A and B'''
	return numpy.sqrt(numpy.sum((A - B)**2))

def nmf(A, W, H, m, n, k):
	'''factorises nonnegative matrix A into W and H
	through multiplicative updates, using euclidian distance
	as a cost function'''
	initdist = distance(numpy.dot(W, H), A)
    
        print()
	for count in range(MAXITER):
		dist = numpy.linalg.norm(numpy.dot(W, H) - A)

		if (dist <= MINERROR):
			return W, H

		print(CLRLINE + "Multiplicative update : %.2f%%"
			% (((count + 1) / float(MAXITER)) * 100))

		WA = numpy.dot(W.T, A)
		WWH = numpy.dot(W.T, numpy.dot(W, H))

		for j in range(m):
			for c in range(k):
				H[c][j] *= (WA[c][j] / WWH[c][j])

		AH = numpy.dot(A, H.T)
		WHH = numpy.dot(numpy.dot(W, H), H.T)

		for i in range(n):
			for c in range(k):
				W[i][c] *= (AH[i][c] / WHH[i][c])
    
	enddist = distance(numpy.dot(W, H), A)
	print("total distance decrease : %.2f (%.2f%%)"
		% ((initdist - enddist), ((initdist - enddist) / initdist) * 100))


def main():
	if (len(sys.argv) != 2):
		usage(sys.argv[0])

	try:
		k = int(sys.argv[1]) # no. of clusters
	except ValueError:
		usage(sys.argv[0])

	A, DF, terms = populate_matrix()

	# apply TF-IDF normalisation
	tf_idf(A, DF)

	A = numpy.array(A)
	n = len(A)         # no. of terms
	m = len(A[0])      # no. of documents

	# randomly initialise W and H
	W = numpy.random.rand(n, k)
	H = numpy.random.rand(k, m)
        nmf(A, W, H, m, n, k)

main()
