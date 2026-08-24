/* startNextQueuedUpload — extracted verbatim from the inline <script> block in
   index.html to keep that block smaller.

   This file is NOT fetched by the browser. Nothing in this app serves static/
   (there is no StaticFiles mount), so a <script src> here would 404. Instead the
   contents below are spliced back into index.html at the exact point the function
   used to occupy, at process startup, by main_live.py and routers/core.py.

   Consequences of that, which must be preserved:
     - This is not an ES module. Do not add import/export statements.
     - The function relies on the surrounding script's top-level bindings
       (globalUploadState, uploadQueue, uploadAbortController, PENDING_JOB_KEY,
       JOB_STATUS_LABELS, currentWorkspaceId, currentWorkspaceTab) and on shared
       helpers (emptyUploadState, parseJSONError, writeStoredJSON,
       removeStoredJSON, renderGlobalActivityBar, renderRecUploadSurface,
       loadLanding, loadWsMeetings, showMeeting, loadWorkspaceTodos). Those stay
       in index.html because they are used throughout it.
     - It assigns to globalUploadState, uploadQueue and uploadAbortController,
       so it must remain in the same script scope as their declarations. */

/* --- Transient-failure policy for the upload POST and the job poller ---
   A momentary 502/503/504 from the ingress (or a dropped connection) must not
   kill an in-flight transcription job: the server-side job keeps running, so
   the client backs off and retries instead of treating it as fatal.
   These are plain function declarations so they share the surrounding script
   scope (and are reachable as window.* for diagnostics). */
function isRetryableUploadStatus(status) {
  var s = Number(status);
  return s === 408 || s === 425 || s === 429 || s === 502 || s === 503 || s === 504;
}

function uploadRetryDelayMs(attempt) {
  var ladder = [3000, 5000, 10000, 20000, 30000];
  var i = Number(attempt);
  if (!isFinite(i) || i < 0) i = 0;
  if (i > ladder.length - 1) i = ladder.length - 1;
  return ladder[Math.floor(i)];
}

var UPLOAD_MAX_POST_RETRIES = 4;   /* initial POST: 5 attempts total */
var UPLOAD_MAX_POLL_FAILURES = 8;  /* poll loop: ~8 consecutive misses before giving up */
var UPLOAD_RETRY_MESSAGE = 'Server temporarily unavailable, retrying…';

async function startNextQueuedUpload() {
  if (globalUploadState.active || uploadQueue.length === 0) return;
  const next = uploadQueue.shift();
  if (!next || !next.file) return;
  const file = next.file;
  const uploadWorkspaceId = next.workspaceId;
  const uploadWorkspaceName = next.workspaceName;
  globalUploadState = {
    active: true,
    fileName: file.name,
    workspaceId: uploadWorkspaceId,
    workspaceName: uploadWorkspaceName,
    step: 'upload',
    stepStartedAt: Date.now(),
    status: 'Saving uploaded file...',
    result: null,
    error: '',
    startedAt: Date.now(),
    completedAt: 0
  };
  renderGlobalActivityBar();
  renderRecUploadSurface();

  const form = new FormData();
  form.append('file', file);
  const url = uploadWorkspaceId ? '/analyze-async?workspace_id=' + uploadWorkspaceId : '/analyze-async';
  uploadAbortController = new AbortController();

  function showRetryNotice(text) {
    globalUploadState.status = text;
    renderGlobalActivityBar();
    if (uploadWorkspaceId === currentWorkspaceId) renderRecUploadSurface();
  }

  try {
    /* Initial POST: retry a few times with backoff on transient 5xx / network drops */
    let resp = null;
    for (let attempt = 0; ; attempt++) {
      let netFailed = false;
      resp = null;
      try {
        resp = await fetch(url, { method: 'POST', body: form, signal: uploadAbortController.signal });
      } catch (netErr) {
        if (netErr && netErr.name === 'AbortError') throw netErr;
        netFailed = true;
      }
      if (netFailed || (resp && isRetryableUploadStatus(resp.status))) {
        if (attempt >= UPLOAD_MAX_POST_RETRIES) {
          if (netFailed) throw new Error('Server temporarily unavailable. Please try again in a moment.');
          throw new Error(await parseJSONError(resp, 'Server temporarily unavailable (' + resp.status + '). Please try again in a moment.'));
        }
        showRetryNotice(UPLOAD_RETRY_MESSAGE);
        await new Promise(function(r) { setTimeout(r, uploadRetryDelayMs(attempt)); });
        if (uploadAbortController && uploadAbortController.signal.aborted) throw new Error('Upload cancelled.');
        continue;
      }
      break;
    }
    if (!resp || !resp.ok) {
      throw new Error(await parseJSONError(resp, 'Server error ' + (resp ? resp.status : '')));
    }
    const queued = await resp.json();
    const jobId = queued.job_id;

    /* Persist the job ID so polling can resume if the page is hard-refreshed */
    writeStoredJSON(PENDING_JOB_KEY, {
      jobId: jobId,
      workspaceId: uploadWorkspaceId,
      workspaceName: uploadWorkspaceName,
      filename: file.name,
      at: Date.now()
    });

    /* Poll /jobs/{id} until done or failed */
    let finalResult = null;
    let pollFailures = 0;
    while (true) {
      await new Promise(function(r) { setTimeout(r, pollFailures ? uploadRetryDelayMs(pollFailures - 1) : 3000); });
      if (uploadAbortController && uploadAbortController.signal.aborted) throw new Error('Upload cancelled.');
      let pollResp = null;
      let pollNetFailed = false;
      try {
        pollResp = await fetch('/jobs/' + jobId);
      } catch (netErr) {
        if (netErr && netErr.name === 'AbortError') throw netErr;
        pollNetFailed = true;
      }
      if (pollNetFailed || (pollResp && isRetryableUploadStatus(pollResp.status))) {
        /* The job is still running server-side — back off and keep checking. */
        pollFailures++;
        if (pollFailures > UPLOAD_MAX_POLL_FAILURES) {
          const transient = new Error('Server temporarily unavailable. Your recording is still processing — reopen this page shortly to see the result.');
          transient.transient = true;
          throw transient;
        }
        showRetryNotice(UPLOAD_RETRY_MESSAGE + ' (attempt ' + pollFailures + ')');
        continue;
      }
      if (!pollResp || !pollResp.ok) throw new Error('Could not check job status.');
      pollFailures = 0;
      const job = await pollResp.json();
      const label = JOB_STATUS_LABELS[job.status] || job.status;
      globalUploadState.status = label;
      renderGlobalActivityBar();
      if (uploadWorkspaceId === currentWorkspaceId) renderRecUploadSurface();
      if (job.status === 'failed') throw new Error(job.error || 'Processing failed.');
      if (job.status === 'done') {
        finalResult = { id: job.meeting_id };
        break;
      }
    }

    if (!finalResult) throw new Error('No result received');
    removeStoredJSON(PENDING_JOB_KEY);
    const hasQueuedItems = uploadQueue.length > 0;
    globalUploadState.active = false;
    globalUploadState.result = hasQueuedItems ? null : finalResult;
    globalUploadState.error = '';
    globalUploadState.status = hasQueuedItems ? 'Saved meeting. Starting next upload...' : 'Analysis complete.';
    globalUploadState.completedAt = Date.now();
    renderGlobalActivityBar();
    if (uploadWorkspaceId === currentWorkspaceId) renderRecUploadSurface();
    if (document.getElementById('landingView').style.display !== 'none') loadLanding();
    if (currentWorkspaceId === uploadWorkspaceId) {
      await loadWsMeetings();
      if (!hasQueuedItems && finalResult && finalResult.id) showMeeting(finalResult.id);
    }
    if (currentWorkspaceId === uploadWorkspaceId && currentWorkspaceTab === 'wsTodosTab') loadWorkspaceTodos();
    uploadAbortController = null;
    if (hasQueuedItems) setTimeout(startNextQueuedUpload, 50);
  } catch (err) {
    const aborted = err && (err.name === 'AbortError' || (uploadAbortController && uploadAbortController.signal.aborted));
    uploadAbortController = null;
    /* Keep the pending-job record on a transient connectivity failure so
       resumePendingUploadJob() can pick the still-running job back up. */
    if (aborted || !(err && err.transient)) removeStoredJSON(PENDING_JOB_KEY);
    if (aborted) {
      uploadQueue = [];
      globalUploadState = emptyUploadState();
      renderGlobalActivityBar();
      renderRecUploadSurface();
      return;
    }
    globalUploadState.active = false;
    globalUploadState.result = null;
    globalUploadState.error = err.message;
    globalUploadState.status = uploadQueue.length ? 'Error on this file — continuing with next upload...' : '';
    globalUploadState.completedAt = Date.now();
    renderGlobalActivityBar();
    if (uploadWorkspaceId === currentWorkspaceId) renderRecUploadSurface();
    if (uploadQueue.length > 0) setTimeout(startNextQueuedUpload, 3000);
  }
}
