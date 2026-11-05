# -*- test-case-name: twisted.web.test.test_reclaim -*-
# Copyright (c) Twisted Matrix Laboratories.
# See LICENSE file for details.

"""
A reactor-bound, host-facing **quiet-reclaim settlement ledger** for the
HTTP/1.1 persistent-connection client.

This module adds no network behaviour of its own.  It gives a host a single
place on a reactor where the *different completion faces* of an HTTP/1.1
request lifecycle are settled as append-only records:

- On L{twisted.web._newclient.HTTP11ClientProtocol}: cancelling a request while
  it is still C{TRANSMITTING} only cancels the request-generation L{Deferred},
  while cancelling in any other state aborts the transport; cancelling
  L{twisted.web.client.readBody} aborts the transport without consulting the
  connection pool's lease at all.  These are distinct faces.
- When a response is fully received, a persistent connection is returned to
  the pool (the quiescent callback runs *before* the response parser is
  disconnected), whereas a C{Connection: close} / non-persistent response, or a
  quiescent callback that raises, abandons / closes the connection without
  pooling it.  These are distinct faces.
- On L{twisted.web.client.HTTPConnectionPool}: putting a connection back into
  the cache, evicting the head of a full per-host queue, losing a connection to
  the idle timeout, and batch-aborting the whole cache (which only joins once
  every abort has completed) are distinct faces.

The ledger is opt-in.  A host L{QuietReclaimLedger.declare declares} one
ledger for a reactor; every L{twisted.web.client.HTTPConnectionPool} created
for that reactor afterwards joins it automatically, as do the protocols and
responses minted by that pool.  When no ledger is declared every instrumented
exit behaves exactly as it did before (the "pin" behaviour): there are no new
required constructor arguments anywhere.

Layering
========

Work that nests or interleaves on the same reactor (an outer request that is
still holding a pool lease or reading a body while an inner request, a body
cancel, an idle timeout or a batch close happens) is bracketed with
L{QuietReclaimLedger.layer}.  A layer is a commit-step boundary:

- settlements made while a layer is open are attributed to that layer and
  cannot be written into an outer or already-committed layer;
- leaving a layer *commits* it.  Nothing is rolled back.  A transport that was
  aborted, a connection that was returned to the pool, or a connection that
  was evicted / batch-aborted stays settled when any ledger layer exits;
- an already-committed layer refuses new settlements, so it cannot be re-opened
  into an "intent window" that pops a previously committed side effect back.

Pool keys
=========

Pool settlements carry the pool key they happened under.  A direct
L{twisted.web.client.Agent} settles under one C{(scheme, host, port)} key per
origin while L{twisted.web.client.ProxyAgent} settles all of its traffic under
the single C{("http-proxy", endpoint)} key.
"""

from __future__ import annotations

import weakref
from contextlib import contextmanager
from types import MappingProxyType
from typing import Any, Iterator, Mapping, NamedTuple, Optional


# ---------------------------------------------------------------------------
# Settlement faces
# ---------------------------------------------------------------------------


#: L{HTTP11ClientProtocol.request} refused a nested request because the
#: protocol was not C{QUIESCENT}; the in-flight lease / generation was not
#: written through.
REQUEST_NOT_SENT = "request-not-sent"

#: A request cancel that only cancelled the request-generation Deferred.
REQUEST_CANCEL_DEFERRED = "request-cancel-deferred-only"

#: A request cancel that itself called C{transport.abortConnection} and
#: disconnected the parser with L{CancelledError}.
REQUEST_CANCEL_ABORT = "request-cancel-abort"

#: A L{twisted.web.client.readBody} cancel that aborted the transport
#: directly, without pool-lease semantics.
READ_BODY_CANCEL_ABORT = "read-body-cancel-abort"

#: The response parser was disconnected; the reason is recorded so a
#: L{CancelledError} parser disconnect can be told apart from a normal
#: L{ConnectionDone} one.
PARSER_DISCONNECT = "parser-disconnect"

#: The response could not be reclaimed (C{Connection: close}, non-persistent
#: request, or response received before the request finished sending) and the
#: connection was given up.
RESPONSE_ABANDON = "response-abandon"

#: A persistent response completed: the quiescent callback returned
#: successfully and the connection became a pool candidate, before the
#: parser was disconnected.
RESPONSE_RECLAIM = "response-quiescent-reclaim"

#: The quiescent callback raised; the connection was lost instead of pooled.
RESPONSE_RECLAIM_FAILED = "response-quiescent-callback-failed"

#: L{HTTP11ClientProtocol.abort} was invoked (the batch-close path): the
#: transport is asked to C{loseConnection} and completion joins when the
#: connection is actually lost.
PROTOCOL_ABORT = "protocol-abort"

#: The pool opened a new connection for a key.
POOL_CONNECT = "pool-new-connection"

#: A cached connection was checked out, popping it from the cache and
#: cancelling its idle timeout.
POOL_LEASE = "pool-lease-checkout"

#: A quiescent connection was accepted into the cache and an idle timeout was
#: scheduled against it.
POOL_PUT = "pool-put"

#: A non-quiescent connection was refused entry to the cache.
POOL_REJECT = "pool-put-rejected-non-quiescent"

#: The head of a full per-host cache queue was evicted: its transport was
#: told to C{loseConnection} and its idle timeout cancelled.
POOL_EVICT = "pool-evict-over-limit"

#: An idle timeout fired: the connection was lost and removed from the cache.
POOL_IDLE = "pool-idle-timeout"

#: One connection was aborted by
#: L{HTTPConnectionPool.closeCachedConnections}.
POOL_CLOSE_ABORT = "pool-close-abort"

#: All connections passed to L{HTTPConnectionPool.closeCachedConnections} have
#: finished losing their connections; the batch-abort join is complete.
POOL_CLOSE_JOIN = "pool-close-join"


#: Faces whose settlement must move the pool-membership view.
_POOL_MEMBERSHIP_FACES = frozenset(
    {POOL_PUT, POOL_EVICT, POOL_IDLE, POOL_CLOSE_ABORT}
)

#: Faces that settle an aborted transport.
_ABORT_FACES = frozenset(
    {
        REQUEST_CANCEL_ABORT,
        READ_BODY_CANCEL_ABORT,
        PROTOCOL_ABORT,
        POOL_CLOSE_ABORT,
    }
)


class Settlement(NamedTuple):
    """
    One append-only completion record.

    @ivar seq: Monotonic sequence number across the whole ledger: causal order
        on the single reactor.
    @ivar face: One of the C{*-face} constants in this module.
    @ivar frame: Identity path of the L{ReclaimLayer} that was current when
        the settlement was committed.
    @ivar key: The pool key the settlement happened under, or L{None} for
        protocol-layer settlements.
    @ivar connection: The L{HTTP11ClientProtocol} the settlement is about, or
        L{None}.
    @ivar generation: The request generation on the connection at settlement
        time, or L{None}.
    @ivar reason: Optional structural reason (for example the parser
        disconnect exception name).
    @ivar detail: Frozen, face-specific extra detail.
    """

    seq: int
    face: str
    frame: tuple[int, ...]
    key: Any
    connection: Any
    generation: Optional[int]
    reason: Optional[str]
    detail: Mapping[str, Any]


class ConnectionSettlement(NamedTuple):
    """
    An immutable snapshot of everything that has irreversibly happened to one
    connection, as observed by the ledger.

    Leaving a ledger layer never changes this snapshot: aborted, reclaimed,
    evicted, idle-closed and batch-closed markers only ever accumulate.
    """

    connection: Any
    generations: tuple[int, ...]
    faces: frozenset[str]
    poolKeys: frozenset[Any]
    aborted: bool
    reclaimed: bool
    abandoned: bool
    evicted: bool
    idleClosed: bool
    batchClosed: bool


class _ReclaimToken:
    """
    Binds an HTTP response (and therefore a L{twisted.web.client.readBody}
    cancel) to the request generation of the connection that produced it.

    The token lets a readBody cancel abort its transport *directly*, bypassing
    pool-lease semantics, while still being settled against the correct
    connection and generation.  Tokens are only minted when a ledger is
    active; responses produced without a ledger carry no token.
    """

    __slots__ = ("ledger", "connection", "generation")

    def __init__(self, ledger: "QuietReclaimLedger", connection: Any, generation: int):
        self.ledger = ledger
        self.connection = connection
        self.generation = generation

    def noteReadBodyAbort(self) -> None:
        """
        Settle a L{READ_BODY_CANCEL_ABORT} face against this token's
        connection and generation.
        """
        self.ledger.settle(
            READ_BODY_CANCEL_ABORT,
            connection=self.connection,
            generation=self.generation,
        )

    def abortConnection(self) -> None:
        """
        Abort the token's underlying transport directly.  No pool lease,
        timeout or cache membership is consulted.
        """
        transport = getattr(self.connection, "transport", None)
        abort = getattr(transport, "abortConnection", None)
        if abort is not None:
            abort()
        elif transport is not None:
            # The body transport may be a proxy that no longer forwards
            # abortConnection; loseConnection is the closest direct
            # equivalent and still deliberately bypasses the pool.
            loseConnection = getattr(transport, "loseConnection", None)
            if loseConnection is not None:
                loseConnection()


class ReclaimLayer:
    """
    A commit-step boundary in a L{QuietReclaimLedger}.

    Settlements committed while the layer is open are attributed to it.  Once
    the layer has been left (committed), it is immutable: attempting to settle
    through it raises L{RuntimeError}.
    """

    __slots__ = ("_ledger", "path", "committed", "_records")

    def __init__(self, ledger: "QuietReclaimLedger", path: tuple[int, ...]):
        self._ledger = ledger
        self.path = path
        self.committed = False
        self._records: list[Settlement] = []

    @property
    def settlements(self) -> tuple[Settlement, ...]:
        """
        The settlements committed directly in this layer (not those of nested
        layers).
        """
        return tuple(self._records)

    def faces(self) -> tuple[str, ...]:
        """
        The faces of the settlements committed directly in this layer, in
        order.
        """
        return tuple(record.face for record in self._records)

    def settle(self, face: str, **detail: Any) -> Optional[Settlement]:
        """
        Settle C{face} against this specific layer.

        @raise RuntimeError: If the layer is no longer the current layer or
            has already been committed.  Reclaim settlements are append-only
            and cannot be popped back into existence.
        """
        ledger = self._ledger
        if not ledger._active or ledger._stack[-1] is not self:
            raise RuntimeError(
                "Cannot settle into reclaim layer %r: the layer is no longer "
                "open.  Settled reclaim effects are append-only and cannot "
                "be re-opened or revoked." % (self.path,)
            )
        return ledger._settle(face, self, **detail)


class QuietReclaimLedger:
    """
    A reactor-bound settlement ledger for HTTP/1.1 persistent-connection
    request cancellation, quiet pool reclamation and pool shutdown.

    Use L{declare} to bind a ledger to a reactor.  All pools (and their
    protocols and responses) created for that reactor afterwards settle into
    it.  Use L{layer} to bracket nested or interleaved work on the reactor.
    """

    _declared: "weakref.WeakKeyDictionary[Any, QuietReclaimLedger]" = (
        weakref.WeakKeyDictionary()
    )

    def __init__(self, reactor: Any) -> None:
        self._reactor = reactor
        self._root = ReclaimLayer(self, ())
        self._stack: list[ReclaimLayer] = [self._root]
        self._log: list[Settlement] = []
        self._seq = 0
        self._frameIds = 0
        # Derived current-membership view.  Only actual settlement events
        # mutate this; leaving a layer never does.
        self._membership: dict[Any, set[Any]] = {}
        self._active = True

    # -- declaration -------------------------------------------------------

    @classmethod
    def declare(cls, reactor: Any) -> "QuietReclaimLedger":
        """
        Declare a settlement ledger on C{reactor}.

        Pools constructed for the same reactor afterwards automatically settle
        into the returned ledger.

        @raise RuntimeError: If an active ledger is already declared for the
            reactor.
        """
        existing = cls._declared.get(reactor)
        if existing is not None and existing._active:
            raise RuntimeError(
                "An active QuietReclaimLedger is already declared for this "
                "reactor; retract it before declaring another."
            )
        ledger = cls(reactor)
        cls._declared[reactor] = ledger
        return ledger

    @classmethod
    def declaredFor(cls, reactor: Any) -> Optional["QuietReclaimLedger"]:
        """
        Return the active ledger declared for C{reactor}, or L{None}.

        A C{None} reactor (pools may be constructed without one until they
        are started) never has a ledger.
        """
        if reactor is None:
            return None
        ledger = cls._declared.get(reactor)
        if ledger is not None and ledger._active:
            return ledger
        return None

    def retract(self) -> None:
        """
        Detach this ledger from its reactor.

        This does *not* undo anything: the full settlement log and snapshots
        remain readable.  New instrumented operations simply stop recording.

        @raise RuntimeError: If a non-root layer is currently open.
        """
        if not self._active:
            return
        if len(self._stack) > 1:
            raise RuntimeError(
                "Cannot retract a QuietReclaimLedger while one of its layers "
                "is still open."
            )
        self._active = False
        self._root.committed = True
        if type(self)._declared.get(self._reactor) is self:
            del type(self)._declared[self._reactor]

    # -- layering ----------------------------------------------------------

    @contextmanager
    def layer(self) -> Iterator[ReclaimLayer]:
        """
        Open a commit-step boundary.

        On exit the layer is committed and every settlement made inside it
        remains in place regardless of how the C{with} block was left; side
        effects are never rolled back when a layer exits.
        """
        if self._active:
            self._frameIds += 1
            frame = ReclaimLayer(
                self, self._stack[-1].path + (self._frameIds,)
            )
            self._stack.append(frame)
            try:
                yield frame
            finally:
                if self._stack[-1] is frame:
                    self._stack.pop()
                frame.committed = True
        else:
            # A retracted ledger still hands out inert layers so instrumented
            # host code keeps working; they record nothing.
            frame = ReclaimLayer(self, (-1,))
            frame.committed = True
            yield frame

    @property
    def rootLayer(self) -> ReclaimLayer:
        """
        The always-open root layer; settlements made outside any explicit
        L{layer} block land here.
        """
        return self._root

    @property
    def active(self) -> bool:
        """
        Whether the ledger is still declared on its reactor.
        """
        return self._active

    # -- settlement --------------------------------------------------------

    def tokenFor(self, connection: Any, generation: int) -> Optional[_ReclaimToken]:
        """
        Mint a reclaim token binding a response / readBody cancel to
        C{connection} and C{generation}, or L{None} when the ledger is no
        longer active.
        """
        if not self._active:
            return None
        return _ReclaimToken(self, connection, generation)

    def settle(
        self,
        face: str,
        *,
        key: Any = None,
        connection: Any = None,
        generation: Optional[int] = None,
        reason: Optional[str] = None,
        **detail: Any,
    ) -> Optional[Settlement]:
        """
        Append one settlement to the ledger, attributed to the current layer.

        Settlements are append-only: this never changes or removes an earlier
        record.
        """
        if not self._active:
            return None
        return self._settle(
            face,
            self._stack[-1],
            key=key,
            connection=connection,
            generation=generation,
            reason=reason,
            **detail,
        )

    def _settle(
        self,
        face: str,
        frame: ReclaimLayer,
        *,
        key: Any = None,
        connection: Any = None,
        generation: Optional[int] = None,
        reason: Optional[str] = None,
        **detail: Any,
    ) -> Settlement:
        record = Settlement(
            self._seq,
            face,
            frame.path,
            key,
            connection,
            generation,
            reason,
            MappingProxyType(dict(detail)),
        )
        self._seq += 1
        frame._records.append(record)
        self._log.append(record)
        if key is not None and connection is not None:
            self._updateMembership(record)
        return record

    def _updateMembership(self, record: Settlement) -> None:
        if record.face == POOL_PUT:
            self._membership.setdefault(record.connection, set()).add(record.key)
        elif record.face in _POOL_MEMBERSHIP_FACES:
            keys = self._membership.get(record.connection)
            if keys is not None:
                keys.discard(record.key)

    # -- observation -------------------------------------------------------

    @property
    def settlements(self) -> tuple[Settlement, ...]:
        """
        All settlements in causal (same-reactor commit) order.
        """
        return tuple(self._log)

    def records(
        self,
        face: Optional[str] = None,
        *,
        connection: Any = None,
        key: Any = None,
        frame: Optional[ReclaimLayer] = None,
        deep: bool = True,
    ) -> tuple[Settlement, ...]:
        """
        Return settlements matching the given filters, in causal order.

        @param face: Only records with this face.
        @param connection: Only records about this connection.
        @param key: Only records under this pool key.
        @param frame: Only records within C{frame}; with C{deep=True} this
            includes records committed in nested layers, with C{deep=False}
            only records committed directly to that frame.
        """
        results = []
        path = None if frame is None else frame.path
        for record in self._log:
            if face is not None and record.face != face:
                continue
            if connection is not None and record.connection is not connection:
                continue
            if key is not None and record.key != key:
                continue
            if path is not None:
                if deep:
                    if record.frame[: len(path)] != path:
                        continue
                elif record.frame != path:
                    continue
            results.append(record)
        return tuple(results)

    def faces(self, **filters: Any) -> tuple[str, ...]:
        """
        Return the faces of L{records} matching C{filters}, in causal order.
        """
        return tuple(record.face for record in self.records(**filters))

    def generations(self, connection: Any) -> tuple[int, ...]:
        """
        Return the request generations observed for C{connection}, in first
        appearance order.
        """
        seen: list[int] = []
        for record in self._log:
            generation = record.generation
            if (
                record.connection is connection
                and generation is not None
                and generation not in seen
            ):
                seen.append(generation)
        return tuple(seen)

    def poolKeys(self) -> frozenset[Any]:
        """
        Return every pool key under which a pool settlement has occurred.
        """
        return frozenset(record.key for record in self._log if record.key is not None)

    def stateOf(self, connection: Any) -> ConnectionSettlement:
        """
        Return an immutable snapshot of the irreversible state settled for
        C{connection}.
        """
        faces = {
            record.face
            for record in self._log
            if record.connection is connection
        }
        return ConnectionSettlement(
            connection=connection,
            generations=self.generations(connection),
            faces=frozenset(faces),
            poolKeys=frozenset(self._membership.get(connection, ())),
            aborted=bool(faces & _ABORT_FACES),
            reclaimed=RESPONSE_RECLAIM in faces,
            abandoned=RESPONSE_ABANDON in faces,
            evicted=POOL_EVICT in faces,
            idleClosed=POOL_IDLE in faces,
            batchClosed=POOL_CLOSE_ABORT in faces,
        )


class _NullReclaimLayer:
    """
    Stand-in L{ReclaimLayer} used when no ledger was declared.
    """

    __slots__ = ()

    path: tuple[int, ...] = ()
    committed = True

    @property
    def settlements(self) -> tuple[Settlement, ...]:
        return ()

    def faces(self) -> tuple[str, ...]:
        return ()

    def settle(self, face: str, **detail: Any) -> None:
        return None


class _NullReclaimLedger:
    """
    No-op ledger used everywhere when the host did not declare one.  Every
    instrumented exit must behave identically to the un-instrumented code
    when this object is in place.
    """

    __slots__ = ()

    active = False

    def settle(self, face: str, **detail: Any) -> None:
        return None

    def records(self, *a: Any, **kw: Any) -> tuple[Settlement, ...]:
        return ()

    def faces(self, *a: Any, **kw: Any) -> tuple[str, ...]:
        return ()

    def tokenFor(self, connection: Any, generation: int) -> None:
        return None

    @contextmanager
    def layer(self) -> Iterator[_NullReclaimLayer]:
        yield _NULL_RECLAIM_LAYER


_NULL_RECLAIM_LEDGER = _NullReclaimLedger()
_NULL_RECLAIM_LAYER = _NullReclaimLayer()
