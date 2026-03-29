import array
import time

class StandardPostings:
    """Uncompressed postings storage using fixed-width 4-byte unsigned integers.

    Assumes the entire postings list for a term fits in memory.
    """

    @staticmethod
    def encode(postings_list: list[int]) -> bytes:
        """Encode a postings list to a byte stream using 4-byte unsigned longs.

        Parameters
        ----------
        postings_list : List[int]
            List of doc IDs (postings).

        Returns
        -------
        bytes
            Raw byte representation of the integer sequence.
        """
        return array.array('L', postings_list).tobytes()

    @staticmethod
    def decode(encoded_postings_list: bytes) -> list[int]:
        """Decode a byte stream back to a postings list.

        Parameters
        ----------
        encoded_postings_list : bytes
            Byte stream produced by encode().

        Returns
        -------
        List[int]
            List of doc IDs.
        """
        decoded_postings_list = array.array('L')
        decoded_postings_list.frombytes(encoded_postings_list)
        return decoded_postings_list.tolist()

    @staticmethod
    def encode_tf(tf_list: list[int]) -> bytes:
        """Encode a list of term frequencies to a byte stream.

        Parameters
        ----------
        tf_list : List[int]
            List of term frequencies.

        Returns
        -------
        bytes
            Raw byte representation of the term frequencies.
        """
        return StandardPostings.encode(tf_list)

    @staticmethod
    def decode_tf(encoded_tf_list: bytes) -> list[int]:
        """Decode a byte stream back to term frequencies.

        Parameters
        ----------
        encoded_tf_list : bytes
            Byte stream produced by encode_tf().

        Returns
        -------
        List[int]
            List of term frequencies.
        """
        return StandardPostings.decode(encoded_tf_list)

class VBEPostings:
    """Postings compression using Variable-Byte Encoding (VBE).

    Stores gap-encoded postings (first doc ID as-is, subsequent entries
    as differences) compressed with VBE. Each integer is split into 7-bit
    chunks; the high bit of the last byte is set to 1 as a terminator.

    Example: [34, 67, 89, 454] -> gaps [34, 33, 22, 365] -> VBE bytes.

    Assumes the entire postings list for a term fits in memory.
    """

    @staticmethod
    def vb_encode_number(number: int) -> bytes:
        """Encode a single non-negative integer using Variable-Byte Encoding.

        Splits the number into 7-bit chunks (big-endian order) and sets the
        high bit of the last byte to 1 as a stop marker.
        """
        bytes = []
        while True:
            bytes.insert(0, number % 128)
            if number < 128:
                break
            number = number // 128
        bytes[-1] += 128
        return array.array('B', bytes).tobytes()

    @staticmethod
    def vb_encode(list_of_numbers: list[int]) -> bytes:
        """Encode a list of non-negative integers using Variable-Byte Encoding."""
        bytes = []
        for number in list_of_numbers:
            bytes.append(VBEPostings.vb_encode_number(number))
        return b"".join(bytes)

    @staticmethod
    def encode(postings_list: list[int]) -> bytes:
        """Encode a sorted postings list into a VBE byte stream.

        Converts to gap-based representation first, then applies VBE.

        Parameters
        ----------
        postings_list : List[int]
            Sorted list of doc IDs (postings).

        Returns
        -------
        bytes
            VBE-encoded byte stream of the gap-based postings.
        """
        gap_postings_list = [postings_list[0]]
        for i in range(1, len(postings_list)):
            gap_postings_list.append(postings_list[i] - postings_list[i-1])
        return VBEPostings.vb_encode(gap_postings_list)

    @staticmethod
    def encode_tf(tf_list: list[int]) -> bytes:
        """Encode a list of term frequencies into a VBE byte stream.

        Parameters
        ----------
        tf_list : List[int]
            List of term frequencies.

        Returns
        -------
        bytes
            VBE-encoded byte stream of the term frequencies.
        """
        return VBEPostings.vb_encode(tf_list)

    @staticmethod
    def vb_decode(encoded_bytestream: bytes) -> list[int]:
        """Decode a VBE-encoded byte stream back to a list of integers."""
        n = 0
        numbers = []
        decoded_bytestream = array.array('B')
        decoded_bytestream.frombytes(encoded_bytestream)
        bytestream = decoded_bytestream.tolist()
        for byte in bytestream:
            if byte < 128:
                n = 128 * n + byte
            else:
                n = 128 * n + (byte - 128)
                numbers.append(n)
                n = 0
        return numbers

    @staticmethod
    def decode(encoded_postings_list: bytes) -> list[int]:
        """Decode a VBE-encoded postings byte stream back to doc IDs.

        Decodes the gap-based list, then reconstructs sorted doc IDs
        by computing prefix sums.

        Parameters
        ----------
        encoded_postings_list : bytes
            Byte stream produced by encode().

        Returns
        -------
        List[int]
            Sorted list of doc IDs.
        """
        decoded_postings_list = VBEPostings.vb_decode(encoded_postings_list)
        total = decoded_postings_list[0]
        ori_postings_list = [total]
        for i in range(1, len(decoded_postings_list)):
            total += decoded_postings_list[i]
            ori_postings_list.append(total)
        return ori_postings_list

    @staticmethod
    def decode_tf(encoded_tf_list: bytes) -> list[int]:
        """Decode a VBE-encoded byte stream back to term frequencies.

        Parameters
        ----------
        encoded_tf_list : bytes
            Byte stream produced by encode_tf().

        Returns
        -------
        List[int]
            List of term frequencies.
        """
        return VBEPostings.vb_decode(encoded_tf_list)
    
class EliasGammaPostings:
    """Postings compression using Elias-Gamma coding.

    Elias-Gamma is a bit-level universal code for positive integers.
    Unlike VBE which operates at byte granularity (7 data bits per byte),
    Elias-Gamma encodes at the bit level, producing tighter output
    especially for small integers which are common in gap-encoded postings.
    Expect better compression ratios than VBE, at the cost of slower
    encode/decode due to bit manipulation overhead.
    """

    @staticmethod
    def eg_encode_number(number: int) -> str:
        """Encode a single positive integer using Elias-Gamma coding.
        Each integer X is encoded as a unary prefix (floor(log2(X)) zeros
        followed by a 1) plus the binary remainder (X - 2^floor(log2(X))
        in floor(log2(X)) bits).

        Parameters
        ----------
        number : int
            A positive integer (>= 1) to encode.

        Returns
        -------
        str
            A string of '0' and '1' characters representing the Elias-Gamma code.
        """
        binary = bin(number)[2:]
        prefix = '0' * (len(binary) - 1)
        return prefix + binary

    @staticmethod
    def eg_encode(list_of_numbers: list[int]) -> bytes:
        """Encode a list of positive integers into a byte stream using Elias-Gamma coding.

        Concatenates all encoded bits and pads to a multiple of 8 with
        trailing zeros. 

        Parameters
        ----------
        list_of_numbers : List[int]
            List of positive integers to encode.

        Returns
        -------
        bytes
            Packed byte stream of the concatenated Elias-Gamma codes.
        """
        bits = []
        for idx, number in enumerate(list_of_numbers):
            if number <= 0:
                raise ValueError(
                    f"Elias-Gamma only supports positive integers, got {number} at index {idx}"
                )
            bits.append(EliasGammaPostings.eg_encode_number(number))

        bitstream = ''.join(bits)
        padding = (8 - len(bitstream) % 8) % 8
        bitstream += '0' * padding

        payload = bytearray()
        for i in range(0, len(bitstream), 8):
            payload.append(int(bitstream[i:i + 8], 2))

        return bytes(payload)

    @staticmethod
    def encode(postings_list: list[int]) -> bytes:
        """Encode a sorted postings list into an Elias-Gamma byte stream.

        Converts the postings list to a gap-based representation first
        (first doc ID kept as-is, subsequent entries stored as differences),
        then encodes the gaps with Elias-Gamma coding.

        Parameters
        ----------
        postings_list : List[int]
            Sorted list of doc IDs (postings).

        Returns
        -------
        bytes
            Elias-Gamma encoded byte stream of the gap-based postings.
        """
        gap_postings_list = [postings_list[0]]
        for i in range(1, len(postings_list)):
            gap_postings_list.append(postings_list[i] - postings_list[i - 1])
        return EliasGammaPostings.eg_encode(gap_postings_list)

    @staticmethod
    def encode_tf(tf_list: list[int]) -> bytes:
        """Encode a list of term frequencies into an Elias-Gamma byte stream.

        Parameters
        ----------
        tf_list : List[int]
            List of term frequencies (each >= 1).

        Returns
        -------
        bytes
            Elias-Gamma encoded byte stream of the term frequencies.
        """
        return EliasGammaPostings.eg_encode(tf_list)

    @staticmethod
    def eg_decode(encoded_bytestream: bytes) -> list[int]:
        """Decode an Elias-Gamma encoded byte stream back to a list of positive integers.

        Converts bytes into a bitstring, then repeatedly decodes Elias-Gamma
        codes: count leading zeros (N), read the next N+1 bits as the integer
        value. Stops when remaining bits can't form a complete code (trailing
        padding is naturally ignored).

        Parameters
        ----------
        encoded_bytestream : bytes
            Byte stream produced by eg_encode.

        Returns
        -------
        List[int]
            The decoded list of positive integers.
        """
        bitstream = ''.join(format(byte, '08b') for byte in encoded_bytestream)

        numbers = []
        pos = 0
        while pos < len(bitstream):
            # Count leading zeros to determine N
            n = 0
            while pos + n < len(bitstream) and bitstream[pos + n] == '0':
                n += 1
            # Read N+1 bits (the '1' + N remainder bits) as the integer
            start = pos + n
            end = start + n + 1
            if end > len(bitstream):
                break
            numbers.append(int(bitstream[start:end], 2))
            pos = end

        return numbers

    @staticmethod
    def decode(encoded_postings_list: bytes) -> list[int]:
        """Decode an Elias-Gamma encoded postings byte stream back to doc IDs.

        Decodes the gap-based list first, then reconstructs the original
        sorted doc IDs by computing prefix sums.

        Parameters
        ----------
        encoded_postings_list : bytes
            Byte stream produced by encode().

        Returns
        -------
        List[int]
            Sorted list of doc IDs.
        """
        gap_list = EliasGammaPostings.eg_decode(encoded_postings_list)
        total = gap_list[0]
        postings_list = [total]
        for i in range(1, len(gap_list)):
            total += gap_list[i]
            postings_list.append(total)
        return postings_list

    @staticmethod
    def decode_tf(encoded_tf_list: bytes) -> list[int]:
        """Decode an Elias-Gamma encoded byte stream back to term frequencies.

        Parameters
        ----------
        encoded_tf_list : bytes
            Byte stream produced by encode_tf().

        Returns
        -------
        List[int]
            List of term frequencies.
        """
        return EliasGammaPostings.eg_decode(encoded_tf_list)


if __name__ == '__main__':
    postings_list = [34, 67, 89, 454, 2345738]
    tf_list = [12, 10, 3, 4, 1]
    for Postings in [StandardPostings, VBEPostings, EliasGammaPostings]:
        print(Postings.__name__)

        t0 = time.time()
        encoded_postings_list = Postings.encode(postings_list)
        encoded_tf_list = Postings.encode_tf(tf_list)
        t1 = time.time()

        print("encoded postings bytes : ", encoded_postings_list)
        print("encoded postings size  : ", len(encoded_postings_list), "bytes")
        print("encoded TF list bytes  : ", encoded_tf_list)
        print("encoded TF list size   : ", len(encoded_tf_list), "bytes")
        print(f"encode time            :  {(t1 - t0) * 1e6:.2f} us")

        decoded_posting_list = Postings.decode(encoded_postings_list)
        decoded_tf_list = Postings.decode_tf(encoded_tf_list)
        print("decoded postings       : ", decoded_posting_list)
        print("decoded TF list        : ", decoded_tf_list)
        assert decoded_posting_list == postings_list, "decoded postings do not match original"
        assert decoded_tf_list == tf_list, "decoded TF list does not match original"
        print()
