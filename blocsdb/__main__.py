import argparse
import gzip
import os
import pathlib
import struct
import sys

from dataclasses import dataclass

def read_gzi(fn):
    with open(fn, "rb") as _in:
        n = struct.unpack("Q", _in.read(8))[0]

        yield 0, 0
        while True:
            try:
                block = struct.unpack("QQ", _in.read(16))
            except struct.error:
                break
            if not block:
                break
            yield block


def convert_blocks(blocks):
    last_block = blocks[0]
    for block in blocks[1:]:
        yield BitBlock(last_block, block)
        last_block = block
    yield BitBlock(last_block,) # None)


@dataclass
class BitBlock:
    starts: tuple = None
    ends: tuple = None

    def contains(self, pos, uncompressed=False):
        return self.starts[uncompressed] <= pos < self.ends[uncompressed]


@dataclass
class SeqDbSegment:
    coords: tuple = None
    start_block: tuple = None
    end_block: tuple = None
    end_block_size: tuple = None
    boundaries: tuple = None
    blocks: tuple = None

    def extract(self, fn, fout):
        if all(self.boundaries):
            raise NotImplementedError("Invalid case: Full file extraction.")
        with open(fn, "rb") as _in:

            if self.boundaries[0]:

                with open(fout, "ab") as _out:
                    offset = 0
                    for blocksize, blocksize_decompressed in self.blocks[1:-1]:
                        _in.seek(offset)
                        _out.write(_in.read(blocksize))
                        offset += blocksize
                    _in.seek(offset)
                with gzip.open(fout, "ab") as _out:
                    _out.write(gzip.decompress(_in.read(self.blocks[-1][0] - self.blocks[-2][0]))[:self.coords[1] - self.blocks[-2][1]])

            elif self.boundaries[1]:

                with gzip.open(fout, "ab") as _out:
                    _in.seek(self.blocks[0][0])
                    _out.write(gzip.decompress(_in.read(self.blocks[1][0]))[self.coords[0] - self.blocks[0][1]:])
                    offset = self.blocks[1][0]
                with open(fout, "ab") as _out:
                    for blocksize, blocksize_decompressed in self.blocks[1:]:
                        _out.write(_in.read(blocksize))
                        offset += blocksize

            else:
                # SeqDbSegment(coords=(519266, 1086382), start_block=(155335, 515288), end_block=(311245, 1032181), end_block_size=(330828, 1097045), boundaries=(False, False), blocks=[(155335, 515288), (311245, 1032181), (330828, 1097045)])

                with gzip.open(fout, "ab") as _out:
                    _in.seek(self.blocks[0][0])
                    _out.write(gzip.decompress(_in.read(self.blocks[1][0] - self.blocks[0][0]))[self.coords[0] - self.blocks[0][1]:])
                    offset = self.blocks[1][0]                                                                                                       
                with open(fout, "ab") as _out:
                    for blocksize, blocksize_decompressed in self.blocks[2:-1]:
                        _in.seek(offset)
                        _out.write(_in.read(blocksize))
                        offset += blocksize
                    _in.seek(offset)
                with gzip.open(fout, "ab") as _out:
                    _out.write(gzip.decompress(_in.read(self.blocks[-1][0] - self.blocks[-2][0]))[:self.coords[1] - self.blocks[-2][1]])



                    

    def extract_old(self, fn, fout):
        with gzip.open(fout, "wb") as _out, open(fn, "rb") as _in:
            if all(self.boundaries):
                raise NotImplementedError("Invalid case: Full file extraction.")
            elif self.boundaries[0]:
                # extract left boundary cluster
                #nbytes_compressed = self.end_block_size[0]
                #_out.write(gzip.decompress(_in.read(nbytes_compressed))[:self.coords[1]])  # don't touch

                offset = 0
                for blocksize, blocksize_decompressed in self.blocks[1:-1]:
                    _in.seek(offset)
                    _out.write(gzip.decompress(_in.read(blocksize)))
                    offset += blocksize
                _in.seek(offset)
                _out.write(gzip.decompress(_in.read(self.blocks[-1][0] - self.blocks[-2][0]))[:self.coords[1] - self.blocks[-2][1]])

            elif self.boundaries[1]:
                # extract right boundary cluster
                #_in.seek(self.start_block[0])
                #start = self.coords[0] - self.start_block[1]
                #_out.write(gzip.decompress(_in.read())[start:])  # don't touch
                _in.seek(self.blocks[0][0])
                _out.write(gzip.decompress(_in.read(self.blocks[1][0]))[self.coords[0] - self.blocks[0][1]:])
                offset = self.blocks[1][0]
                for blocksize, blocksize_decompressed in self.blocks[1:]:
                    _out.write(gzip.decompress(_in.read(blocksize)))
                    offset += blocksize
                    
            else:
                # extract inner cluster
                #_in.seek(self.start_block[0])
                #d = _in.read(self.end_block_size[0] - self.start_block[0])
                #_out.write(gzip.decompress(d)[self.coords[0] - self.start_block[1]:self.coords[1] - self.start_block[1]])

                _in.seek(self.blocks[0][0])
                _out.write(gzip.decompress(_in.read(self.blocks[1][0] - self.blocks[0][0]))[self.coords[0] - self.blocks[0][1]:])
                offset = self.blocks[1][0]
                for blocksize, blocksize_decompressed in self.blocks[2:-1]:
                    _in.seek(offset)
                    _out.write(gzip.decompress(_in.read(blocksize)))
                    offset += blocksize
                _in.seek(offset)
                _out.write(gzip.decompress(_in.read(self.blocks[-1][0] - self.blocks[-2][0]))[:self.coords[1] - self.blocks[-2][1]])
                




SeqDbSegment(coords=(0, 519266), start_block=(0, 0), end_block=(155335, 515288), end_block_size=(174759, 579666), boundaries=(True, False), blocks=[(0, 0), (155335, 515288), (174759, 579666)])
SeqDbSegment(coords=(519266, 1086382), start_block=(155335, 515288), end_block=(311245, 1032181), end_block_size=(330828, 1097045), boundaries=(False, False), blocks=[(155335, 515288), (311245, 1032181), (330828, 1097045)])
SeqDbSegment(coords=(1086382, 1499424), start_block=(311245, 1032181), end_block=(441265, 1481902), end_block_size=None, boundaries=(False, True), blocks=[(311245, 1032181), (441265, 1481902)])

def find_segment(blocks, start, end):
    start_block, end_block = None, None
    last_block = 0, 0
    boundary_start, boundary_end = False, False

    boundary_start = start == 0
    next_right = None
    segment_blocks = set()

    for i, block in enumerate(blocks):

        if start_block and end_block:
            next_right = last_block
            segment_blocks.add(last_block)
            break

        if start_block is None and start < block[1]:
            start_block = last_block
            segment_blocks.add(last_block)
        if end_block is None and end < block[1]:
            end_block = last_block
            segment_blocks.add(last_block)

        last_block = block



    if end_block is None:
        boundary_end = True
        end_block = block
        segment_blocks.add(block)

    return SeqDbSegment((start, end), start_block, end_block, next_right, (boundary_start, boundary_end), sorted(segment_blocks))


def extract(seqdb_file, gzi_file, fai_file, group_id, outfile):
    blocks = list(read_gzi(gzi_file))
    # specI_v4_00000	0	1618186374                                 
    with open(fai_file, "rt") as _in:
        fai_d = {
            line.split("\t")[0]: tuple(map(int, line.strip().split("\t")[1:]))
            for line in _in
        } 
    coords = fai_d.get(group_id)
    if coords is None:
        raise ValueError(f"Cannot find {group_id=}")
    offset, size = coords
 
    segment = find_segment(blocks, offset, offset + size)
    segment.extract(seqdb_file, outfile)




def main():

    ap = argparse.ArgumentParser()
    ap.add_argument("seqdb", type=str, help="Path to a blocsdb database.")
    ap.add_argument("group", type=str, help="ID of sequence group to extract.")
    ap.add_argument("outfile", type=str, help="Filename to write extracted sequence group.")
    args = ap.parse_args()

    seqdb_file = f"{args.seqdb}.bgzf"
    if not pathlib.Path(seqdb_file).is_file():
        raise ValueError(f"{seqdb_file} does not exist.")

    gzi_file = f"{seqdb_file}.gzi"
    if not pathlib.Path(gzi_file).is_file():
        raise ValueError(f"{gzi_file} does not exist.")
    
    fai_file = f"{seqdb_file}.cindex"
    if not pathlib.Path(fai_file).is_file():
        raise ValueError(f"{fai_file} does not exist.")
    
    extract(seqdb_file, gzi_file, fai_file, args.outfile)
    return None

    



    """
    seqdb_file = sys.argv[1]
    gzi_file = sys.argv[2]      

    blocks = list(read_gzi(gzi_file))

    segment = find_segment(blocks, 0, 519266)
    segment.extract(seqdb_file, "segment1b.fa.gz")
    print(segment)

    segment = find_segment(blocks, 1086382, 1086382 + 413042)
    segment.extract(seqdb_file, "segment3b.fa.gz")
    print(segment)

    segment = find_segment(blocks, 519266, 519266 + 567116)
    segment.extract(seqdb_file, "segment2b.fa.gz")
    print(segment)
    return None
    """

    seqdb_file = sys.argv[1]
    gzi_file = sys.argv[2]
    fai_file = sys.argv[3]
    cluster = sys.argv[4]

    extract(seqdb_file, gzi_file, fai_file, cluster)


    return None



if __name__ == "__main__":
    main()

