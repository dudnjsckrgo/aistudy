import argparse
import logging

logger = logging.getLogger('process-confirm')
logger.setLevel(logging.DEBUG)
fh = logging.FileHandler('process.log')
fh.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.ERROR)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
ch.setFormatter(formatter)
# add the handlers to the logger
logger.addHandler(fh)
logger.addHandler(ch)

parser = argparse.ArgumentParser()
parser.add_argument('--verbose', '-v',
    action='store_true',
    help='verbose flag' )
parser.add_argument('--foo', nargs='?', const='c', default='d')
parser.add_argument('bar', nargs='?', default='d')
parser.parse_args(['XX', '--foo', 'YY'])
parser.parse_args(['XX', '--foo'])
parser.parse_args([])
args = parser.parse_args()
print(args)
if args.verbose:
    print("~ Verbose!")
else:
    print("~ Not so verbose")