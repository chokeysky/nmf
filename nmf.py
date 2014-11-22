#!/usr/bin/python2.7
#
# COMP41450 Assigment 2
# NMF implementation with sparse term-document matrix
# Erik Nyquist 13206065

import numpy
import sys

TERMS="bbcnews.terms"
TDMATRIX="bbcnews.mtx"


def populate_matrix():
	'''reads in a list of terms & a term-document matrix
	from supplied files, populates an array.
	returns: A, terms
	A = term-document matrix in the form of 2D array of floats.
	terms = the terms as a 1D array of strings.'''

	with open(TDMATRIX, "r") as mfh:
		header = mfh.readline() #skip header

		#read matrix dimensions
		rows, columns, numvalues = mfh.readline().split()

		#initialise matrix A with above dimensions to all zeros
		A = [[0 for i in range(int(rows))] for j in range(int(columns))]

		#populate matrix A
		count = 0
		for line in mfh:
			term, doc, freq = line.split()
			A[int(doc) - 1][int(term) - 1] = float(freq)
			count += 1

		#make sure our entry count matches the number of non-zero
		#values given on the 2nd line
		if int(numvalues) != count:
			print("Malformed term-document matrix in file '" + TDMATRIX + "'\n"
				"Expecting " + numvalues + " entries but found " + str(count))
			sys.exit(-1)
	with open(TERMS, "r") as tfh:
		terms =	tfh.readlines()

	return A, terms	

def main():
	A, terms = populate_matrix()

main()
