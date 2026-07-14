///|
#external
type WebNNProgramScheduler

///|
extern "js" fn new_program_scheduler(pool_size : Int) -> WebNNProgramScheduler =
  #| (poolSize) => {
  #|   const scheduler = {
  #|     poolSize,
  #|     available: Array.from({ length: poolSize }, (_, index) => index),
  #|     jobs: [],
  #|     active: 0,
  #|     maximumConcurrency: 0,
  #|     closed: false,
  #|     disposer: undefined,
  #|     disposed: false,
  #|   };
  #|   const disposeIfReady = () => {
  #|     if (scheduler.closed && scheduler.active === 0 &&
  #|         scheduler.jobs.length === 0 && !scheduler.disposed) {
  #|       scheduler.disposed = true;
  #|       scheduler.disposer();
  #|     }
  #|   };
  #|   scheduler.pump = () => {
  #|     while (scheduler.available.length > 0 && scheduler.jobs.length > 0) {
  #|       const slot = scheduler.available.shift();
  #|       const job = scheduler.jobs.shift();
  #|       scheduler.active += 1;
  #|       scheduler.maximumConcurrency = Math.max(
  #|         scheduler.maximumConcurrency,
  #|         scheduler.active,
  #|       );
  #|       Promise.resolve()
  #|         .then(() => job.operation(slot))
  #|         .then(job.resolve, job.reject)
  #|         .finally(() => {
  #|           scheduler.active -= 1;
  #|           scheduler.available.push(slot);
  #|           scheduler.pump();
  #|           disposeIfReady();
  #|         });
  #|     }
  #|     disposeIfReady();
  #|   };
  #|   return scheduler;
  #| }

///|
extern "js" fn enqueue_program_run(
  scheduler : WebNNProgramScheduler,
  operation : (Int) -> @js.Promise[Array[WebNNNamedValues]],
) -> @js.Promise[Array[WebNNNamedValues]] =
  #| (scheduler, operation) => {
  #|   if (scheduler.closed) {
  #|     return Promise.reject(new Error("WebNN program has been destroyed"));
  #|   }
  #|   return new Promise((resolve, reject) => {
  #|     scheduler.jobs.push({ operation, resolve, reject });
  #|     scheduler.pump();
  #|   });
  #| }

///|
extern "js" fn enqueue_program_destroy(
  scheduler : WebNNProgramScheduler,
  operation : () -> Unit,
) -> Unit =
  #| (scheduler, operation) => {
  #|   if (scheduler.closed) return;
  #|   scheduler.closed = true;
  #|   scheduler.disposer = operation;
  #|   scheduler.pump();
  #| }

///|
extern "js" fn program_maximum_concurrency(
  scheduler : WebNNProgramScheduler,
) -> Int =
  #| (scheduler) => scheduler.maximumConcurrency
