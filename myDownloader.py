import asyncio
import concurrent.futures
import requests
import os

#https://stackoverflow.com/questions/58571343/downloading-a-large-file-in-parts-using-multiple-parallel-threads

URL = 'https://file-examples.com/wp-content/uploads/2017/04/file_example_MP4_1920_18MG.mp4'
OUTPUT = 'video.mp4'


    
async def get_size(url):
    response = requests.head(url)
    size = int(response.headers['Content-Length'])
    return size


def download_range(url, start, end, output):
    headers = {'Range': f'bytes={start}-{end}'}
    response = requests.get(url, headers=headers)

    with open(output, 'wb') as f:
        for part in response.iter_content(1024):
            f.write(part)


async def download(executor, url, output, chunk_size=1000000):
    loop = asyncio.get_event_loop()

    file_size = await get_size(url)
    chunks = range(0, file_size, chunk_size)

    tasks = [
        loop.run_in_executor(
            executor,
            download_range,
            url,
            start,
            start + chunk_size - 1,
            f'{output}.part{i}',
        )
        for i, start in enumerate(chunks)
    ]

    await asyncio.wait(tasks)

    with open(output, 'wb') as o:
        for i in range(len(chunks)):
            chunk_path = f'{output}.part{i}'

            with open(chunk_path, 'rb') as s:
                o.write(s.read())

            os.remove(chunk_path)


if __name__ == '__main__':
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=3)
    loop = asyncio.get_event_loop()

    try:
        loop.run_until_complete(
            download(executor, URL, OUTPUT)
        )
    finally:
        loop.close()