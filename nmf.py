#!/usr/bin/python2.7
#
# COMP41450 Assigment 2
# Topic modelling of a term-document matrix
# through nonnegative matrix factorisation
# Erik Nyquist 13206065
#
# requires following input files to be defined:
# - term-document matrix stored in Matrix Market format (*.mtx)
# - a newline-seperated list of the terms (*.terms)
#
# takes 1 optional command-line argument, K (integer),
# which represents the number of clusters (topics).
# default value of K is 5.

import numpy
import sys
import os

# input files
TERMS="bbcnews.terms"
TDMATRIX="bbcnews.mtx"

# this implementation of NMF takes a long time to converge,
# however 100 iterations is enough to see some meaningful
# results and completes in a reasonable time. Increase MAXITER
# if you have some free time :)
MAXITER=100

MINERROR=0.02
NUMTERMS=10 # no. of terms to show for each cluster

# VT100 terminal control codes
# for clearing the last console line
CLRLINE='\x1b[2K'
UP1='\x1b[1A'

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
	'''takes matrix A and performs TF-IDF normalisation
	an excellent practical example can be seen at :
	http://en.wikipedia.org/wiki/Tf-idf'''
	n = len(A)
        print()
	for x in range(len(A)):
		for y in range(len(A[x])):
			if (A[x][y] != 0):
				A[x][y] *= numpy.log10(n / DF[y])
        	print(UP1 + CLRLINE + "Normalising input matrix : %.2f%%"
			% (((x + 1) / float(n) * 100)))
            
def distance(A, B):
	'''returns the euclidian distance between A and B'''
	return numpy.sqrt(numpy.sum((A - B)**2))

def nmf(A, W, H, m, n, k):
	'''factorises nonnegative matrix A into W and H
	through multiplicative updates, using euclidian distance
	as a cost function'''
	initdist = distance(numpy.dot(W, H), A)
    	print("Initial distance : %.2f" % (initdist))
        print('\n')
	for count in range(MAXITER):
		dist = distance(numpy.dot(W, H), A)

		if (dist <= MINERROR):
			return W, H

		print(UP1 + CLRLINE + UP1 + CLRLINE +
			"Multiplicative update : %.2f%%\n"
			"Distance from convergence : %.2f"
			% (((count + 1) / float(MAXITER)) * 100, dist))

		# update H from WA / WWH
		WA = numpy.dot(W.T, A)
		WWH = numpy.dot(W.T, numpy.dot(W, H))

		for j in range(m):
			for c in range(k):
				H[c][j] *= (WA[c][j] / WWH[c][j])

		# update W from AH / WHH
		AH = numpy.dot(A, H.T)
		WHH = numpy.dot(numpy.dot(W, H), H.T)

		for i in range(n):
			for c in range(k):
				W[i][c] *= (AH[i][c] / WHH[i][c])
    
	enddist = distance(numpy.dot(W, H), A)
	print("total distance decrease : %.2f (%.2f%%)"
		% ((initdist - enddist), ((initdist - enddist) / initdist) * 100))

def show_top_terms(W, H, m, n, k, terms):
	'''displays the top NUMTERMS terms for each cluster, k'''
	for c in range(k):
		# populate a dict with term:membership
		# as a key:value pair for each term in this cluster
		toptermsd = {}
		for t in range(len(W)):
			toptermsd[terms[t]] = W[t][c]

		# sort the terms into a list of tuples, ordered by value
		# (cluster membership)
		topterms = sorted(toptermsd.items(), key=lambda x: x[1])
		toptermsd.clear()

		# print the last NUMTERMS terms
		print("\nCluster %d:" % (c + 1))
		for j in range(1, NUMTERMS + 1):
			print("\t" + str(topterms[-j][0]).strip()
				+ " (%.2f)" % (topterms[-j][1]))
			

def main():
	if (len(sys.argv) > 2):
		usage(sys.argv[0])

	elif (len(sys.argv) == 2):
		try:
			k = int(sys.argv[1])
		except ValueError:
			usage(sys.argv[0])

	else:
		k = 5

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
	show_top_terms(W, H, m, n, k, terms)

main()
