# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "chess",
#     "pyarrow>=19.0.0",
#     "numpy",
#     "rbloom", # Bloom filter for fast deduplication
#       ]
# ///

from asyncio.log import logger
import logging
import random
import subprocess
import sys
from typing import override
import chess
import chess.engine
import chess.pgn
from pathlib import Path
import re
import pyarrow as pa
import pyarrow.dataset as pa_ds
from rbloom import Bloom

# match Stockfish_*_64-bit.commented.*.pgn.7z

pattern = re.compile(r".*.pgn.zst")


def get_games(search_path: Path) -> list[Path]:
    paths = [p for p in search_path.iterdir() if pattern.match(p.name)]
    return paths


def stream_games_pipe(archive_path):
    """Stream through games using unzst"""
    
    process = subprocess.Popen(
        ['zstd', '-d', '-c', str(archive_path)],  # -d for decompress, -c for stdout
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        bufsize=4,  # Line buffered
        universal_newlines=True  # This handles the text decoding for us
    )
    
    try:
        while True:
            try:
                game = chess.pgn.read_game(process.stdout)
                if game is None:
                    break
                yield game
            except Exception as e:
                logging.warning(f"Error reading game: {e}", exc_info=True)
                continue
    finally:
        process.stdout.close()
        process.wait()


class AdmeteFeaturesCommand(chess.engine.BaseCommand[None]):
    def __init__(self, engine: chess.engine.UciProtocol) -> None:
        super().__init__(engine)
        self.engine =  engine

    def start(self) -> None:
        self.engine.send_line("features quiece")

    @override
    def line_received(self, line: str) -> None:
        self.result.set_result(line)
        self.set_finished()

class AdmeteHeuristicCommand(chess.engine.BaseCommand[None]):
    def __init__(self, engine: chess.engine.UciProtocol) -> None:
        super().__init__(engine)
        self.engine =  engine

    def start(self) -> None:
        self.engine.send_line("h")
    
    @override
    def line_received(self, line: str) -> None:
        # score is just an integer
        self.result.set_result(int(line))
        self.set_finished()

def iter_games(archive_paths):
    for archive_path in archive_paths:
        print(f"Reading {archive_path}")
        yield from stream_games_pipe(archive_path)

def iter_games_filtered(archive_paths, min_elo=3000):
    for game in iter_games(archive_paths):
        if game is None or game.headers.get('WhiteElo') is None or game.headers.get('BlackElo') is None:
            continue
        elo_white = int(game.headers.get('WhiteElo'))
        elo_black = int(game.headers.get('BlackElo'))
        if min(elo_white, elo_black) >= min_elo:
            yield game

from pathlib import Path
import re
import pyarrow as pa
import numpy as np
import random
from pathlib import Path
import chess.engine
import logging
import asyncio
import queue
import threading

logger = logging.getLogger(__name__)

import queue
import threading
import chess.engine
import chess

class PositionSolverThreadPool:
    def __init__(self, engine_path: Path, num_threads: int):
        self.engine_path = engine_path
        self.num_threads = num_threads
        self.threads = []
        self.queue = queue.Queue(maxsize=1000)
        self.results = queue.Queue(maxsize=1000)
        self.stop_event = threading.Event()
        self.running = False
        
    def worker(self):
        async def run_work():
            transport, engine = await chess.engine.popen_uci(self.engine_path)
            board = chess.Board()
            while not self.stop_event.is_set():
                try:
                    fen, meta = self.queue.get(timeout=5)
                    board.set_fen(fen)
                except queue.Empty:
                    continue
                try:
                    engine._position(board) 
                    features = await engine.communicate(AdmeteFeaturesCommand)
                    self.results.put((features, meta))
                except chess.engine.EngineTerminatedError:
                    logger.warning("Engine terminated", exc_info=True)
                    break
                finally:
                    self.queue.task_done()
            await engine.quit()
            
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(run_work())
        except Exception as e:
            logger.error(f"Error in worker thread: {e}", exc_info=True)
        finally:
            loop.close()
            self.stop_event.set()  # Ensure the stop event is set to signal termination
        # Not closing the loop here, as it can cause issues
        
    def start(self):
        assert not self.running, "Thread pool is already running"
        self.stop_event.clear()
        for _ in range(self.num_threads):  # Fixed syntax
            thread = threading.Thread(target=self.worker)
            thread.start()
            self.threads.append(thread)
        self.running = True  # Set running to True when started
        
    def stop(self):
        assert self.running, "Thread pool is not running"
        self.stop_event.set()
        while not self.queue.empty():
            self.queue.get()
            self.queue.task_done()
        for thread in self.threads:
            thread.join()
        self.threads = []  # Clear threads list
        self.running = False
        
    def submit(self, fen, meta):
        if not self.running:
            self.start()
        self.queue.put((fen, meta))


def process_positions(paths, engine_path: Path, elo:int=0, positions:int|None = None, batch_size:int=10000):
    """Exposes an iterator that yields pyarrow tables of training data."""
    pattern = re.compile(r"\[%eval ([+-]?\d+\.\d+)\]")
    random.seed(42)
    assert isinstance(positions, int), "positions must be an integer"

    result_map = {
        "1-0": "1",
        "0-1": "0",
        "1/2-1/2": "D",
        "*": "U"
    }

    runner = PositionSolverThreadPool(engine_path, num_threads=4)

    def producer():
        seen_positions = Bloom(positions, 0.1)
        for game in iter_games_filtered(paths, elo):
            board = game.board()
            result = result_map.get(game.headers.get('Result'), "U")
            m_count = 0
            for node in game.mainline():
                if runner.stop_event.is_set():
                    return
                try:
                    board.push(node.move)
                except Exception as e:
                    logger.warning(f"Error pushing move {node.move} to board: {e}", exc_info=True)
                    break
                # Skip if no evaluation in comment
                match = pattern.search(node.comment)
                if match is None:
                    break
                eval = int(float(match.group(1))*100)
                if node.next() is None:
                    break # Skip last node, not useful
                if m_count < 10:
                    m_count += 1
                    continue
                # number of pieces on the board
                n_pieces = len(board.piece_map())
                if n_pieces <= 7:
                    continue  # Skip tablebase positions
                if board.fen() in seen_positions:
                    continue
                seen_positions.add(board.fen())
                meta = (board.fen(), eval, result, board.turn)
                runner.submit(board.fen(), meta)
    thread = threading.Thread(target=producer)
    runner.start()
    thread.start()

    feature_type = pa.list_(pa.int8(), 64)
    f_batch = []
    sf_eval_batch = []
    result_batch = []
    set_type_batch = []

    def make_table():
        table = pa.RecordBatch.from_arrays([
            pa.array(f_batch, type=feature_type),
            pa.array(sf_eval_batch, type=pa.int32()),
            pa.array(result_batch, type=pa.string()),
            pa.array(set_type_batch, type=pa.int8())
            ], names=['features', 'sf_eval', 'result', 'set_type'])
        f_batch.clear()
        sf_eval_batch.clear()
        result_batch.clear()
        set_type_batch.clear()
        return table

    counter = 0
    while True:
        if runner.stop_event.is_set():
            return
        try:
            features, meta = runner.results.get(timeout=5)
        except queue.Empty:
            continue
        runner.results.task_done()
        # get the capturing group match
        fen, eval, result, turn = meta
        if turn == chess.WHITE:
            eval = eval
            result_r = result
        else:
            eval = -eval
            if result == "1":
                result_r = "0"
            elif result == "0":
                result_r = "1"
            else:
                result_r = result

        # Convert string feature vectors to actual lists of integers
        f_vector = pa.scalar([int(x, 16) for x in features], type=feature_type)
        
        # Always add to batch (no train/test split here - we'll handle that when consuming the iterator)
        f_batch.append(f_vector)
        sf_eval_batch.append(eval)
        result_batch.append(result_r)
        r = random.random()
        if r < 0.8:
            set_type_batch.append(0)
        elif r < 0.9:
            set_type_batch.append(1)
        else:
            set_type_batch.append(2)
        counter += 1
        if counter % batch_size == 0:
            print(f"Processed {counter//1000:>10}K positions")   
            yield make_table()    
            
        if positions is not None and counter >= positions:
            print(f"Processed {counter} positions, reached maximum.")
            yield  make_table()
            break
    runner.stop()
    thread.join()
    

def process_positions_and_save(paths, output_path: Path, engine_path: Path, elo:int=0, positions:int|None = None):
    # Extract feature vector length
    feature_length = 64
    
    iter = process_positions(paths, engine_path, elo=elo, positions=positions)

    # Define PyArrow schema with fixed-size lists for feature vectors
    schema = pa.schema([
        pa.field('features', pa.list_(pa.int8(), feature_length)),
        pa.field('sf_eval', pa.int32()),
        pa.field('result', pa.string()),
        pa.field('set_type', pa.int8()), # 0 = training, 1 = test, 2 = validation
    ])

    partition = pa_ds.partitioning(pa.schema([
        pa.field('set_type', pa.int8()),
    ]))

    file_options = pa_ds.ParquetFileFormat().make_write_options(compression="none") # we're cpu bound on decoding, so leave compression off
    # also this is ~ 50 GB so who cares, admitedly, the sparse data compresses really nicely

    pa_ds.write_dataset(
        iter,
        output_path,
        format="parquet",
        schema=schema,
        partitioning=partition,
        file_options=file_options,
        existing_data_behavior="delete_matching",
        use_threads=True,
    )

# package to parse command line arguments
import argparse

async def main() -> None:
    arg_parser = argparse.ArgumentParser(description="Process and save positions from pgn files.")
    arg_parser.add_argument("games", type=str, help="Path to the folder containing the compressed pgn files.")
    arg_parser.add_argument("output", type=str, help="Path to the output file.")
    arg_parser.add_argument("engine", type=str, help="Path to the engine executable.")
    arg_parser.add_argument("elo", type=int, default=None, help="Minimum elo to include a game")
    arg_parser.add_argument("positions", type=int, default=None, help="Games to include")

    args = arg_parser.parse_args()

    paths = get_games(Path(args.games))
    output_path = Path(args.output)
    engine_path = Path(args.engine)
    assert engine_path.exists(), f"Engine path {engine_path} does not exist."
    assert output_path.exists(), f"Output path {output_path} does not exist."
    assert output_path.is_dir(), f"Output path {output_path} is not a directory."
    assert len(paths) > 0, f"No games found in {args.games}"
    process_positions_and_save(
        paths,
        output_path / "data",
        engine_path,
        elo=args.elo,
        positions=args.positions
    )

from asyncio import run

if __name__ == "__main__":

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
           logging.StreamHandler(sys.stdout),
        ],
    )
    logger.setLevel(logging.INFO)
    run(main())
