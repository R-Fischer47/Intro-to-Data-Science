# This is the object that contains the LSA semantic space.
#
# This is where the magic takes place.
#
# Revision History:
# ~~~~~~~~~~~~~~~~~
# 08/10/2019 - CJL
###

import numpy as np  # For SVD Calculation
import math  # For sqrt function


class SemanticSpace:
    ##
    # Constructor
    #
    # Takes one parameter, I, which is the InvertedIndex that we want to build
    # a semantic space from.
    #
    # @param I - InvertedIndex object.
    #
    # Revision History:
    # ~~~~~~~~~~~~~~~~~
    # 08/10/2019 - Created (CJL).
    ###
    def __init__(self, I):
        # Store the Inverted Index
        self.I = I

        # Generate our term by document matrix "A"
        self.A = self.I.generate_term_by_doc_matrix()

        # Output from our SVD operation
        self.T = None
        self.S = None
        self.Dt = None

        # Perform the SVD operation on A
        self.T, self.S, self.Dt = np.linalg.svd(self.A, full_matrices=False)

        # Create a prefabricated inverse of the singular values as well
        # as the squares.
        # (shortcut for operations - can you see how they would help?)
        self.S_inv = [1.0 / x for x in self.S]
        self.S_sq = [x * x for x in self.S]

        # Set our dimensional reduction value default (value of "k")
        self.max_dimension = 2

    ##
    # Create a query vector from a string of text representing the
    # query.  This includes the "folding in" of the query such that
    # it is in the same dimensional space as our semantic space
    # created through our SVD calculation.
    #
    # Note, this essentially transforms our query into a "Document"
    # in our semantic space.  In theory, if we were expanding our
    # semantic space with more documents we would just "fold them
    # in" with an operation like this and append them to the Dt
    # matrix.  Obviously the more you fold in the more it waters
    # down the relationships encapsulated in the original SVD
    # calculation.
    #
    # @param q - String representing query.
    #
    # @return Vector representing the query in our semantic space.
    #
    # Revision History:
    # ~~~~~~~~~~~~~~~~~
    # 09/10/2019 - Created (CJL).
    ###
    def create_query_vector(self, q):
        ret_q = [0.0 for i in range(self.I.get_total_terms())]

        # This is where we ensure the query text is tokenized and processed
        # the same as the text was in the Inverted Index (we use the same
        # function call).  It's important any terms we introduce as a query
        # are in the same format as what can be found in the semantic space
        # (if they are there at all).
        q = self.I.process_text(q)

        # You will need to:
        #    A) create a query vector (simple vector representing
        #       the TF of each of the terms present in the query (processed
        #       above).
        #    B) Complete the "fold_in_query" method

        # Now create a query vector
        terms = [x[0] for x in self.I.terms]
        for t in q:
            for y in range(len(terms)):
                if t == terms[y]:
                    ret_q[y] = 1

        ##
        # If our semantic space was generated from an initial weighted A matrix
        # than we would need to weight our query vector the same way.  You'd
        # do that here.
        ##

        ret_q = self.fold_in_query(ret_q)

        return ret_q

    ##
    # This function takes a vector representation of a document or query
    # and folds it into our semantic space.
    #
    # The new query is calculated as qTS^{-1}
    #
    # @param q - Query in a vector format.
    #
    # @return Query folded into the semantic space.
    #
    # Revision History:
    # ~~~~~~~~~~~~~~~~~
    # 09/10/2019 - Created (CJL).
    ###
    def fold_in_query(self, q):
        # new query is qTS^{-1}
        no_terms = self.T.shape[0]
        rank = self.T.shape[1]

        # Create vector for folded in query
        fol_q = [0.0 for i in range(rank)]

        # The T matrix will have as many rows as corresponds to terms but
        # will have *rank* number of columns.  Perform the calculation
        # as outlined above.
        #
        # Can you optimise this further using the S_inv array in the constructor?

        # Multiply q by T and scale by S^{-1}
        fol_q = [0.0 for i in range(rank)]
        for i in range(rank):
            for j in range(no_terms):
                fol_q[i] += q[j] * self.T[j][i]
            fol_q[i] = fol_q[i] * self.S_inv[i]

        return fol_q

    ##
    # Calculates the cosine between a query vector and a document
    # in our semantic space.
    #
    # NOTE:  This assumes the query vector has already been "folded in"
    #        to the semantic space.
    #
    # @param q   - Query vector
    #        doc - Index into the Dt indicating the column (document)
    #              we wish to compare the query against.
    #
    # @return The cosine value between the query and specified document
    #         vector.
    #
    # Revision History:
    # ~~~~~~~~~~~~~~~~~
    # 09/10/2019 - Created (CJL).
    ###
    def cosine_with_doc(self, q, doc):
        # m_q and m_d are the magnitudes of the query and document vectors
        m_q = 0
        m_d = 0
        for i in range(self.max_dimension):
            m_q += q[i]*q[i]*self.S_sq[i]
            m_d += self.Dt[i][doc]*self.Dt[i][doc]*self.S_sq[i]

        m_q = math.sqrt(m_q)
        m_d = math.sqrt(m_d)
        # This would be the dot product of q*S [dot] Dt[*]*S[doc]
        calc = 0.0
        # Can you optimise this further using the S_sq array generated
        # in the constructor?

        # q*S [dot] Dt[*]*S[doc]
        for i in range(self.max_dimension):
            calc += q[i]*self.Dt[i][doc]*self.S_sq[i]
        cos = calc/(m_q*m_d)

        return cos

    ##
    # Calculates the cosine between two terms in our semantic space.
    #
    # @param t1 - Index (row) of the first term to compare
    # @param t2 - Index (row) of the 2nd term to compare.
    #
    # @return The cosine value term t1 and term t2 in our semantic space.
    #
    # Revision History:
    # ~~~~~~~~~~~~~~~~~
    # 09/10/2019 - Created (CJL).
    ###
    def cosine_with_term(self, t1, t2):
        # m_t1 and m_t2 are the magnitudes of the term vectors
        t1 = self.T[t1]
        t2 = self.T[t2]
        m_t1 = 0
        m_t2 = 0
        for i in range(self.max_dimension):
            m_t1 += t1[i]*t1[i]*self.S_sq[i]
            m_t2 += t2[i]*t2[i]*self.S_sq[i]

        m_t1 = math.sqrt(m_t1)
        m_t2 = math.sqrt(m_t2)
        # This would be the dot product of t1*S [dot] t2*S
        calc = 0.0
        # Can you optimise this further using the S_sq array generated
        # in the constructor?

        # t1*S [dot] t2*S[doc]
        for i in range(self.max_dimension):
            calc += t1[i]*t2[i]*self.S_sq[i]
        cos = calc/(m_t1*m_t2)

        return cos
