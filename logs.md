
Personal

Containers
llm-retrieval-system-conversation-service-1

llm-retrieval-system-conversation-service-1
721feb40eeca
llm-retrieval-system-conversation-service:latest
8001:8001⁠
STATUS
Running (52 minutes ago)


INFO:     172.19.0.5:36908 - "GET /metrics HTTP/1.1" 404 Not Found

/usr/local/lib/python3.11/site-packages/starlette/datastructures.py:626: RuntimeWarning: coroutine 'EnhancedRAGPipeline.get_pipeline_stats' was never awaited

  for idx, (item_key, item_value) in enumerate(self._list):

RuntimeWarning: Enable tracemalloc to get the object allocation traceback

INFO:     127.0.0.1:47494 - "GET /health HTTP/1.1" 200 OK

INFO:     162.159.140.245:51665 - "POST /api/v1/customers/test_customer_123/profile HTTP/1.1" 500 Internal Server Error

ERROR:    Exception in ASGI application

Traceback (most recent call last):

  File "/app/src/api/routes.py", line 236, in update_customer_profile

    updated_profile = await rag_pipeline.customer_profile_manager.update_customer_profile(

                            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

AttributeError: 'CustomerProfileManager' object has no attribute 'update_customer_profile'


During handling of the above exception, another exception occurred:


Traceback (most recent call last):

  File "/usr/local/lib/python3.11/site-packages/uvicorn/protocols/http/httptools_impl.py", line 426, in run_asgi

    result = await app(  # type: ignore[func-returns-value]

             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

  File "/usr/local/lib/python3.11/site-packages/uvicorn/middleware/proxy_headers.py", line 84, in __call__

    return await self.app(scope, receive, send)

           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

  File "/usr/local/lib/python3.11/site-packages/fastapi/applications.py", line 1106, in __call__

    await super().__call__(scope, receive, send)

  File "/usr/local/lib/python3.11/site-packages/starlette/applications.py", line 122, in __call__

    await self.middleware_stack(scope, receive, send)

  File "/usr/local/lib/python3.11/site-packages/starlette/middleware/errors.py", line 184, in __call__

    raise exc

  File "/usr/local/lib/python3.11/site-packages/starlette/middleware/errors.py", line 162, in __call__

    await self.app(scope, receive, _send)

  File "/usr/local/lib/python3.11/site-packages/starlette/middleware/cors.py", line 83, in __call__

    await self.app(scope, receive, send)

  File "/usr/local/lib/python3.11/site-packages/starlette/middleware/exceptions.py", line 79, in __call__

    raise exc

  File "/usr/local/lib/python3.11/site-packages/starlette/middleware/exceptions.py", line 68, in __call__

    await self.app(scope, receive, sender)

  File "/usr/local/lib/python3.11/site-packages/fastapi/middleware/asyncexitstack.py", line 20, in __call__

    raise e

  File "/usr/local/lib/python3.11/site-packages/fastapi/middleware/asyncexitstack.py", line 17, in __call__

    await self.app(scope, receive, send)

  File "/usr/local/lib/python3.11/site-packages/starlette/routing.py", line 718, in __call__

    await route.handle(scope, receive, send)

  File "/usr/local/lib/python3.11/site-packages/starlette/routing.py", line 276, in handle

    await self.app(scope, receive, send)

  File "/usr/local/lib/python3.11/site-packages/starlette/routing.py", line 66, in app

    response = await func(request)

               ^^^^^^^^^^^^^^^^^^^

  File "/usr/local/lib/python3.11/site-packages/fastapi/routing.py", line 274, in app

    raw_response = await run_endpoint_function(

                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^

  File "/usr/local/lib/python3.11/site-packages/fastapi/routing.py", line 191, in run_endpoint_function

    return await dependant.call(**values)

           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

  File "/app/src/api/routes.py", line 247, in update_customer_profile

    logger.error(f"Profile update failed: {e}")

    ^^^^^^

NameError: name 'logger' is not defined

INFO:     172.19.0.5:40646 - "GET /metrics HTTP/1.1" 404 Not Found

INFO:     172.19.0.5:42380 - "GET /metrics HTTP/1.1" 404 Not Found

INFO:     172.19.0.9:48614 - "GET /health HTTP/1.1" 200 OK

INFO:     172.19.0.5:47894 - "GET /metrics HTTP/1.1" 404 Not Found


