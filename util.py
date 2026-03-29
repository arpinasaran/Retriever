class IdMap:
    """Bidirectional mapping between strings (terms or document paths) and integer IDs.

    Internally uses a dict for O(1) string → ID lookup and a list for O(1)
    ID → string lookup.

    Examples
    --------
    >>> m = IdMap()
    >>> m["hello"]   # assigns and returns ID 0
    0
    >>> m["world"]   # assigns and returns ID 1
    1
    >>> m[0]         # reverse lookup
    'hello'
    """

    def __init__(self):
        self.str_to_id = {}
        self.id_to_str = []

    def __len__(self):
        """Return the number of entries in the map."""
        return len(self.id_to_str)

    def __get_str(self, i):
        """Return the string associated with integer ID i."""
        return self.id_to_str[i]

    def __get_id(self, s):
        """Return the integer ID for string s, assigning a new one if absent."""
        if s not in self.str_to_id:
            self.id_to_str.append(s)
            self.str_to_id[s] = len(self.id_to_str) - 1
        return self.str_to_id[s]

    def __getitem__(self, key):
        """Look up by string (returns/assigns an ID) or by int (returns the string).

        Parameters
        ----------
        key : str or int
            String to look up (or auto-assign) an ID for, or an int to reverse-look up.
        """
        if type(key) is int:
            return self.__get_str(key)
        elif type(key) is str:
            return self.__get_id(key)
        else:
            raise TypeError

def sorted_merge_posts_and_tfs(posts_tfs1, posts_tfs2):
    """Merge two sorted (doc_id, tf) lists, accumulating TFs for shared doc IDs.

    Parameters
    ----------
    posts_tfs1 : List[Tuple[int, int]]
        First sorted list of (doc_id, tf) pairs.
    posts_tfs2 : List[Tuple[int, int]]
        Second sorted list of (doc_id, tf) pairs.

    Returns
    -------
    List[Tuple[int, int]]
        Merged sorted list; TFs are summed for doc IDs that appear in both lists.

    Examples
    --------
    >>> sorted_merge_posts_and_tfs([(1,34),(3,2),(4,23)], [(1,11),(2,4),(4,3),(6,13)])
    [(1, 45), (2, 4), (3, 2), (4, 26), (6, 13)]
    """
    i, j = 0, 0
    merge = []
    while (i < len(posts_tfs1)) and (j < len(posts_tfs2)):
        if posts_tfs1[i][0] == posts_tfs2[j][0]:
            freq = posts_tfs1[i][1] + posts_tfs2[j][1]
            merge.append((posts_tfs1[i][0], freq))
            i += 1
            j += 1
        elif posts_tfs1[i][0] < posts_tfs2[j][0]:
            merge.append(posts_tfs1[i])
            i += 1
        else:
            merge.append(posts_tfs2[j])
            j += 1
    while i < len(posts_tfs1):
        merge.append(posts_tfs1[i])
        i += 1
    while j < len(posts_tfs2):
        merge.append(posts_tfs2[j])
        j += 1
    return merge

def test(output, expected):
    """ simple function for testing """
    return "PASSED" if output == expected else "FAILED"

if __name__ == '__main__':

    doc = ["halo", "semua", "selamat", "pagi", "semua"]
    term_id_map = IdMap()
    assert [term_id_map[term] for term in doc] == [0, 1, 2, 3, 1], "term_id salah"
    assert term_id_map[1] == "semua", "term_id salah"
    assert term_id_map[0] == "halo", "term_id salah"
    assert term_id_map["selamat"] == 2, "term_id salah"
    assert term_id_map["pagi"] == 3, "term_id salah"

    docs = ["/collection/0/data0.txt",
            "/collection/0/data10.txt",
            "/collection/1/data53.txt"]
    doc_id_map = IdMap()
    assert [doc_id_map[docname] for docname in docs] == [0, 1, 2], "docs_id salah"

    assert sorted_merge_posts_and_tfs([(1, 34), (3, 2), (4, 23)], \
                                      [(1, 11), (2, 4), (4, 3 ), (6, 13)]) == [(1, 45), (2, 4), (3, 2), (4, 26), (6, 13)], "sorted_merge_posts_and_tfs salah"
