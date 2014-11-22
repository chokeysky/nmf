#!/usr/bin/python2.7
#
# COMP41450 Assigment 2
# NMF implementation with sparse term-document matrix
# Erik Nyquist 13206065

import numpy
import sys

TERMS="bbcnews.terms"
TDMATRIX="bbcnews.mtx"

def malformed(msg):
	'''prints message & exits in the case of a malformed
	term-document matrix'''

	print("Malformed term-document matrix in file '" + TDMATRIX + "' :\n"
		+ msg)
	sys.exit(-1)

def populate_matrix():
	'''reads in a list of terms & a term-document matrix
	from supplied files, populates an array.
	returns: A, terms
	A = term-document matrix in the form of 2D array of floats.
	terms = the terms as a 1D array of strings.'''

	with open(TERMS, "r") as tfh:
		terms =	tfh.readlines()

	with open(TDMATRIX, "r") as mfh:
		header = mfh.readline() # skip header

		# read matrix dimensions
		rows, columns, numvalues = mfh.readline().split()

		# initialise matrix A with above dimensions to all zeros
		A = [[0.0 for i in range(int(rows))] for j in range(int(columns))]

		# populate matrix A
		count = 0
		for line in mfh:
			term, doc, freq = line.split()
			term = int(term)
			doc = int(doc)
			freq = float(freq)

			# sanity check:
			# double entry means I did something wrong,
			# or matrix was generated incorrectly
			if A[doc -1][term - 1] == 0:
				A[doc - 1][term - 1] = freq
				count += 1
			else:
				malformed("Entry #" + str(count + 1) + ", for term '" +
					terms[term - 1].strip() + "' in document #" + str(doc) +
					"\nalready has a value of " + A[doc -1][term -1] +
					".\nEntries cannot be assigned twice.")

		# final sanity check
		if int(numvalues) != count:
				malformed("Expecting " + numvalues +
					" entries but found " + str(count))

	return A, terms	

def main():
	A, terms = populate_matrix()
	A = numpy.array(A)

main()
