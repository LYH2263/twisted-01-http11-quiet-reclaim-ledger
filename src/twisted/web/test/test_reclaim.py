# Copyright (c) Twisted Matrix Laboratories.
# See LICENSE file for details.

"""
Tests for the host-facing HTTP/1.1 quiet-reclaim settlement ledger
(L{twisted.web._reclaim.QuietReclaimLedger}).

Every test here drives the *same* reactor (a L{twisted.internet.task.Clock}
plus L{twisted.internet.testing.StringTransport}s) through real
L{HTTP11ClientProtocol} / L{HTTPConnectionPool} / L{Agent} / L{ProxyAgent}
objects.  The assertions follow causal timing (which completion face settled
before which, and which transport / lease / timeout actually moved) rather
than merely the length of a pool dictionary or a protocol C{_state} string.
"""

from twisted.internet import protocol
from twisted.internet.defer import (
    CancelledError,
    Deferred,
    succeed,
)
from twisted.internet.error import ConnectionDone
from twisted.internet.task import Clock
from twisted.internet.testing import StringTransport
from twisted.python.failure import Failure
from twisted.trial.unittest import TestCase
from twisted.web import _reclaim
from twisted.web._newclient import (
    HTTP11ClientProtocol,
    Request,
    RequestGenerationFailed,
    RequestNotSent,
    Response,
    ResponseNeverReceived,
)
from twisted.web.client import (
    Agent,
    HTTPConnectionPool,
    ProxyAgent,
    QuietReclaimLedger,
    readBody,
)
from twisted.web.http_headers import Headers
from twisted.web.iweb import IAgentEndpointFactory, IBodyProducer
from twisted.internet.interfaces import IStreamClientEndpoint
from zope.interface import implementer


# A complete, immediate 200 response with no body.
EMPTY_200 = b"HTTP/1.1 200 OK\r\nContent-Length: 0\r\n\r\n"

# Response headers announcing a six byte body, followed by only half of it.
PARTIAL_200 = b"HTTP/1.1 200 OK\r\nContent-Length: 6\r\n\r\nabc"

CLOSING_200 = (
    b"HTTP/1.1 200 OK\r\n"
    b"Content-Length: 0\r\n"
    b"Connection: close\r\n\r\n"
)


class FakeRequest:
    """
    Minimal stand-in for L{twisted.web._newclient.Request}.
    """

    method = b"GET"

    def __init__(self, persistent=True, method=b"GET"):
        self.persistent = persistent
        self.method = method

    def writeTo(self, transport):
        transport.write(b"%s / HTTP/1.1\r\nHost: example\r\n\r\n" % (self.method,))
        return succeed(None)


@implementer(IBodyProducer)
class PausedProducer:
    """
    A body producer whose producing L{Deferred} never fires until it is
    cancelled, so the protocol stays in C{TRANSMITTING} (exactly like the
    pinned test suite's mid-production request).
    """

    def __init__(self, length=10):
        self.length = length
        self.consumer = None
        self.finished = None
        self.cancelled = False
        self.stopped = False

    def startProducing(self, consumer):
        self.consumer = consumer

        def canceller(deferred):
            # The write Deferred cancellation reaches the producer; the
            # Deferred then errbacks with CancelledError, as a real
            # cancelled production would.
            self.cancelled = True
            deferred.errback(Failure(CancelledError()))

        self.finished = Deferred(canceller)
        return self.finished

    def pauseProducing(self):
        pass

    def resumeProducing(self):
        pass

    def stopProducing(self):
        self.stopped = True


_POST_HEADERS = Headers({b"Host": [b"example.invalid"]})


def pausedPostRequest(producer):
    """
    A real POST C{Request} whose declared body is still being produced.
    """
    return Request(b"POST", b"/bar", _POST_HEADERS, producer)


def makeProtocol(ledger=None, quiescentCallback=None):
    """
    Build an L{HTTP11ClientProtocol} attached to a fresh L{StringTransport}.
    """
    if quiescentCallback is None:
        quiescentCallback = lambda c: None
    protocol = HTTP11ClientProtocol(
        quiescentCallback,
        reclaimLedger=ledger,
    )
    transport = StringTransport()
    protocol.makeConnection(transport)
    return protocol, transport


def resultOf(d):
    """
    Extract the already-fired result of C{d} synchronously.
    """
    results = []
    d.addBoth(results.append)
    assert len(results) == 1, "Deferred had not fired yet"
    result = results[0]
    if isinstance(result, Failure):
        result.raiseException()
    return result


def failureOf(d):
    """
    Extract the already-fired L{Failure} of C{d} synchronously.
    """
    results = []
    d.addBoth(results.append)
    assert len(results) == 1, "Deferred had not fired yet"
    assert isinstance(results[0], Failure)
    return results[0]


class CapturingEndpoint:
    """
    An endpoint that builds real protocols from a pool factory and remembers
    every L{(protocol, transport)} pair connected.
    """

    def __init__(self, pool):
        self._pool = pool
        self.connections = []

    def connect(self, factory):
        protocol = factory.buildProtocol(None)
        transport = StringTransport()
        protocol.makeConnection(transport)
        self.connections.append((protocol, transport))
        return succeed(protocol)


def rawProtocol(checkout):
    """
    Return the raw L{HTTP11ClientProtocol} of a checkout, which is either the
    protocol itself (a fresh connection) or a retry wrapper (a cached one).
    """
    return getattr(checkout, "_clientProtocol", checkout)


def completeResponse(pool, endpoint, key, request=None):
    """
    Run one full request/response cycle through C{pool} on C{endpoint}.

    @return: The raw L{HTTP11ClientProtocol}, its transport and the response
        L{Deferred} result.
    """
    if request is None:
        request = FakeRequest()
    checkout = resultOf(pool.getConnection(key, endpoint))
    raw = rawProtocol(checkout)
    result = checkout.request(request)
    responses = []
    result.addCallback(responses.append)
    raw.dataReceived(EMPTY_200)
    assert len(responses) == 1
    return raw, raw.transport, result


class LedgerDeclarationTests(TestCase):
    def setUp(self):
        self.clock = Clock()

    def test_poolWithoutDeclarationUsesNullLedger(self):
        """
        With no declared ledger a pool behaves exactly as before: no new
        constructor argument is needed and settlement is inert.
        """
        pool = HTTPConnectionPool(self.clock)
        self.assertIs(pool._reclaimLedger, _reclaim._NULL_RECLAIM_LEDGER)
        # The null ledger swallows every call.
        self.assertIsNone(pool._reclaimLedger.settle("anything", key=1))
        self.assertEqual(pool._reclaimLedger.faces(), ())

    def test_declareBindsPoolProtocolAndResponse(self):
        """
        A pool created after declaration joins the ledger; protocols minted
        by the pool's factory carry the ledger, and responses produced by
        those protocols carry a reclaim token for readBody cancels.
        """
        ledger = QuietReclaimLedger.declare(self.clock)
        self.addCleanup(ledger.retract)
        pool = HTTPConnectionPool(self.clock)
        self.assertIs(pool._reclaimLedger, ledger)

        endpoint = CapturingEndpoint(pool)
        checkout = resultOf(pool.getConnection((b"http", b"x", 80), endpoint))
        raw = rawProtocol(checkout)
        self.assertIs(raw._reclaimLedger, ledger)

        result = checkout.request(FakeRequest())
        responses = []
        result.addCallback(responses.append)
        raw.dataReceived(PARTIAL_200)
        response = responses[0]
        self.assertIsNotNone(response._reclaimToken)
        self.assertIs(response._reclaimToken.connection, raw)
        self.assertEqual(response._reclaimToken.generation, 1)

    def test_duplicateDeclarationRejected(self):
        ledger = QuietReclaimLedger.declare(self.clock)
        self.addCleanup(ledger.retract)
        self.assertRaises(
            RuntimeError, QuietReclaimLedger.declare, self.clock
        )

    def test_retractIsNotUndo(self):
        """
        Retracting detaches the ledger but keeps its log and snapshots;
        further settlements become no-ops and a new declaration is allowed.
        """
        ledger = QuietReclaimLedger.declare(self.clock)
        protocol, transport = makeProtocol(ledger)
        result = protocol.request(FakeRequest())
        result.cancel()
        failureOf(result).trap(Exception)
        self.assertNotEqual(ledger.settlements, ())

        ledger.retract()
        self.assertIsNone(QuietReclaimLedger.declaredFor(self.clock))
        # History survives retraction.
        self.assertNotEqual(ledger.settlements, ())
        self.assertTrue(ledger.stateOf(protocol).aborted)
        # Settlements after retraction are inert.
        self.assertIsNone(ledger.settle(_reclaim.POOL_PUT))
        # Declaring again now works.
        second = QuietReclaimLedger.declare(self.clock)
        second.retract()

    def test_retractRefusesOpenLayer(self):
        ledger = QuietReclaimLedger.declare(self.clock)
        with ledger.layer():
            self.assertRaises(RuntimeError, ledger.retract)
        ledger.retract()


class LedgerLayeringTests(TestCase):
    def setUp(self):
        self.clock = Clock()
        self.ledger = QuietReclaimLedger.declare(self.clock)
        self.addCleanup(self.ledger.retract)

    def test_innerAndOuterDoNotCross(self):
        protocol, transport = makeProtocol(self.ledger)
        result = protocol.request(FakeRequest())
        # A settle outside any explicit layer belongs to the root layer.
        rootFaces = []
        with self.ledger.layer() as outer:
            rootFaces.append(None)
            with self.ledger.layer() as inner:
                nested = protocol.request(FakeRequest())
                failureOf(nested).trap(RequestNotSent)
                self.assertEqual(
                    inner.faces(), (_reclaim.REQUEST_NOT_SENT,)
                )
            # Inner layer committed; its settlement is not visible in the
            # outer layer's direct view.
            self.assertEqual(outer.faces(), ())
            # The inner frame is immutable.
            self.assertTrue(inner.committed)
            self.assertRaises(
                RuntimeError, inner.settle, _reclaim.POOL_PUT
            )
            # ... while the outer layer can still settle.
            self.ledger.settle(
                _reclaim.RESPONSE_RECLAIM, connection=protocol
            )
            self.assertEqual(
                outer.faces(), (_reclaim.RESPONSE_RECLAIM,)
            )
        # The outer request never completed; tearing it down settles on the
        # request-abort face after both layers committed.
        result.cancel()
        failureOf(result).trap(Exception)

        # Global log keeps every record, ordered by frame depth and sequence.
        records = self.ledger.records(
            face=_reclaim.REQUEST_NOT_SENT, deep=False, frame=inner
        )
        self.assertEqual(len(records), 1)
        # Deep view from the outer layer includes the nested record.
        outerDeep = self.ledger.records(frame=outer, deep=True)
        self.assertIn(records[0], outerDeep)

    def test_committedEffectsSurviveLayerExit(self):
        """
        An abort committed in an inner layer stays settled after every layer
        exits: nothing is popped back or revoked.
        """
        protocol, transport = makeProtocol(self.ledger)
        result = protocol.request(FakeRequest())
        with self.ledger.layer():
            with self.ledger.layer():
                result.cancel()
        failureOf(result).trap(Exception)
        self.assertTrue(transport.disconnected)
        self.assertTrue(self.ledger.stateOf(protocol).aborted)
        faces = self.ledger.faces(connection=protocol)
        self.assertEqual(
            faces,
            (
                _reclaim.REQUEST_CANCEL_ABORT,
                _reclaim.PARSER_DISCONNECT,
            ),
        )


class ProtocolCancelForkTests(TestCase):
    """
    The request-cancel Deferred-only face, the request-cancel abort face and
    the readBody abort face are different completion faces, all on one reactor
    and one ledger.
    """

    def setUp(self):
        self.clock = Clock()
        self.ledger = QuietReclaimLedger.declare(self.clock)
        self.addCleanup(self.ledger.retract)

    def test_cancelWhileTransmittingOnlyCancelsDeferred(self):
        """
        Cancelling while TRANSMITTING only cancels the request generation's
        write Deferred: the cancel itself never calls abortConnection and
        never disconnects the parser.  The write-generation errback then
        fails the request with RequestGenerationFailed (and is what aborts
        the transport), a different completion face from the explicit
        cancel-abort fork.
        """
        protocol, transport = makeProtocol(self.ledger)
        producer = PausedProducer()
        result = protocol.request(pausedPostRequest(producer))
        self.assertEqual(protocol._state, "TRANSMITTING")

        result.cancel()

        # The write generation was cancelled and the request fails through
        # the generation-failure path ...
        failure = failureOf(result)
        failure.trap(RequestGenerationFailed)
        self.assertTrue(failure.value.reasons[0].check(CancelledError))
        self.assertEqual(protocol._state, "GENERATION_FAILED")
        # The cancellation was forwarded all the way to the producer.
        self.assertTrue(producer.cancelled)
        # ... and although the generation-failure path aborts the raw
        # transport, the cancel itself settled only the deferred-only face:
        # no cancel-abort and no parser disconnect with CancelledError.
        self.assertEqual(
            self.ledger.faces(connection=protocol),
            (_reclaim.REQUEST_CANCEL_DEFERRED,),
        )
        record = self.ledger.records(
            face=_reclaim.REQUEST_CANCEL_DEFERRED
        )[0]
        self.assertEqual(record.generation, 1)
        self.assertEqual(record.detail["state"], "TRANSMITTING")
        self.assertFalse(self.ledger.stateOf(protocol).aborted)
        self.assertEqual(
            self.ledger.records(face=_reclaim.PARSER_DISCONNECT), ()
        )

    def test_cancelAfterFullResponseWhileTransmittingStillDeferredOnly(self):
        """
        When the whole response finishes while the request is still writing,
        the protocol enters TRANSMITTING_AFTER_RECEIVING_RESPONSE and gives
        the connection up on the abandon face (loseConnection, never a pool).
        A cancel landing *in that window* - the response Deferred has been
        chained to the request result but has not fired yet - still takes the
        Deferred-only face: it forwards the cancellation to the producer and
        never aborts the transport.
        """
        # The TRANSMITTING_AFTER_RECEIVING_RESPONSE cancel window exists only
        # inside the finisher call stack; loseConnection (invoked by _giveUp
        # in that window) is the host-reachable seam into it.
        class CancelOnLoseTransport(StringTransport):
            _cancelDeferred = None
            _cancelArmed = False

            def arm(self, deferred):
                self._cancelDeferred = deferred
                self._cancelArmed = True

            def loseConnection(self):
                if self._cancelArmed:
                    self._cancelArmed = False
                    self._cancelDeferred.cancel()
                super().loseConnection()

        protocol = HTTP11ClientProtocol(reclaimLedger=self.ledger)
        transport = CancelOnLoseTransport()
        protocol.makeConnection(transport)
        producer = PausedProducer()
        result = protocol.request(pausedPostRequest(producer))
        transport.arm(result)
        protocol.dataReceived(EMPTY_200)

        # The cancel landed in the TARR window: cancellation was forwarded
        # to the producer ...
        self.assertTrue(producer.cancelled)
        # ... and the request result failed through cancellation ...
        failureOf(result).trap(CancelledError)
        # ... but the transport was only ever loseConnection()'d by the
        # abandon path; the cancel itself did not abort it.
        self.assertTrue(transport.disconnecting)
        self.assertFalse(transport.disconnected)
        self.assertEqual(
            self.ledger.faces(connection=protocol),
            (
                _reclaim.RESPONSE_ABANDON,
                _reclaim.REQUEST_CANCEL_DEFERRED,
                _reclaim.PARSER_DISCONNECT,
            ),
        )
        deferredRecord = self.ledger.records(
            face=_reclaim.REQUEST_CANCEL_DEFERRED
        )[0]
        self.assertEqual(
            deferredRecord.detail["state"],
            "TRANSMITTING_AFTER_RECEIVING_RESPONSE",
        )
        # The racing response delivery against the already-cancelled chained
        # result only logs (AlreadyCalledError / CancelledError); force GC so
        # the failure is observed here and now rather than on a later test.
        import gc

        gc.collect()
        self.flushLoggedErrors()

    def test_cancelWhileWaitingAbortsAndDisconnectsWithCancelledError(self):
        """
        Cancelling after the request was written aborts the transport and
        disconnects the parser with CancelledError; the request fails.
        """
        protocol, transport = makeProtocol(self.ledger)
        result = protocol.request(FakeRequest())
        self.assertEqual(protocol._state, "WAITING")

        result.cancel()

        self.assertTrue(transport.disconnected)
        failure = failureOf(result)
        failure.trap(ResponseNeverReceived)
        self.assertTrue(
            failure.value.reasons[0].check(CancelledError)
        )
        faces = self.ledger.faces(connection=protocol)
        self.assertEqual(
            faces,
            (_reclaim.REQUEST_CANCEL_ABORT, _reclaim.PARSER_DISCONNECT),
        )
        disconnect = self.ledger.records(
            face=_reclaim.PARSER_DISCONNECT
        )[0]
        self.assertEqual(disconnect.reason, "CancelledError")
        self.assertTrue(self.ledger.stateOf(protocol).aborted)

    def test_deferredOnlyAndAbortAreSeparateOnSameLedger(self):
        """
        Two protocols sharing the reactor/ledger settle distinct faces that
        do not bleed into each other.
        """
        slowProtocol, slowTransport = makeProtocol(self.ledger)
        slowResult = slowProtocol.request(pausedPostRequest(PausedProducer()))
        slowResult.cancel()
        failureOf(slowResult).trap(RequestGenerationFailed)

        abortProtocol, abortTransport = makeProtocol(self.ledger)
        abortResult = abortProtocol.request(FakeRequest())
        abortResult.cancel()
        failureOf(abortResult).trap(ResponseNeverReceived)

        # The raw transports both end up closed, but only the WAITING cancel
        # settled a cancel-abort face: at the ledger level the two forks never
        # bleed into each other.
        self.assertTrue(slowTransport.disconnected)
        self.assertTrue(abortTransport.disconnected)
        slowState = self.ledger.stateOf(slowProtocol)
        abortState = self.ledger.stateOf(abortProtocol)
        self.assertEqual(
            slowState.faces, frozenset({_reclaim.REQUEST_CANCEL_DEFERRED})
        )
        self.assertFalse(slowState.aborted)
        self.assertEqual(
            abortState.faces,
            frozenset(
                {_reclaim.REQUEST_CANCEL_ABORT, _reclaim.PARSER_DISCONNECT}
            ),
        )
        self.assertTrue(abortState.aborted)

    def test_nestedRequestDoesNotWriteThroughInFlightGeneration(self):
        """
        A request attempted while the protocol is busy is refused on the
        not-sent face without bumping the generation.
        """
        protocol, transport = makeProtocol(self.ledger)
        first = protocol.request(FakeRequest())
        nested = protocol.request(FakeRequest())
        failureOf(nested).trap(RequestNotSent)
        self.assertEqual(self.ledger.generations(protocol), (1,))
        self.assertEqual(
            self.ledger.faces(connection=protocol),
            (_reclaim.REQUEST_NOT_SENT,),
        )
        # The first request is still the owner of the connection.
        self.assertEqual(protocol._state, "WAITING")
        first.cancel()
        failureOf(first).trap(Exception)

    def test_readBodyCancelAbortsDirectlyAndForksFromRequestCancel(self):
        """
        Cancelling readBody settles the read-body abort face against the
        request generation that owns the connection, aborts the transport
        directly and never touches pool-lease accounting; it is a different
        face from a protocol request cancel.
        """
        # A pool on the same reactor proves no pool face is settled by a
        # readBody cancel.
        pool = HTTPConnectionPool(self.clock, persistent=True)
        endpoint = CapturingEndpoint(pool)
        key = (b"http", b"example", 80)
        checkout = resultOf(pool.getConnection(key, endpoint))
        protocol = rawProtocol(checkout)
        transport = protocol.transport

        result = checkout.request(FakeRequest())
        responses = []
        result.addCallback(responses.append)
        protocol.dataReceived(PARTIAL_200)
        response = responses[0]
        self.assertEqual(response._reclaimToken.generation, 1)

        body = readBody(response)

        def poolFaces():
            return [
                face
                for face in self.ledger.faces()
                if face.startswith("pool-")
            ]

        poolFacesBefore = poolFaces()
        body.cancel()
        failureOf(body).trap(CancelledError)

        # The underlying transport was aborted directly ...
        self.assertTrue(transport.disconnected)
        # ... on the read-body face ...
        bodyRecords = self.ledger.records(
            face=_reclaim.READ_BODY_CANCEL_ABORT
        )
        self.assertEqual(len(bodyRecords), 1)
        self.assertIs(bodyRecords[0].connection, protocol)
        self.assertEqual(bodyRecords[0].generation, 1)
        # ... without any parser disconnect or pool settlement at all.
        self.assertEqual(
            self.ledger.faces(connection=protocol),
            (_reclaim.READ_BODY_CANCEL_ABORT,),
        )
        self.assertEqual(poolFacesBefore, poolFaces())

        # A fresh request cancel on another connection is the request-abort
        # face, never confused with the read-body face.
        other, otherTransport = makeProtocol(self.ledger)
        otherResult = other.request(FakeRequest())
        otherResult.cancel()
        failureOf(otherResult).trap(Exception)
        self.assertEqual(
            self.ledger.records(face=_reclaim.READ_BODY_CANCEL_ABORT, connection=other),
            (),
        )
        self.assertIn(
            _reclaim.REQUEST_CANCEL_ABORT,
            self.ledger.stateOf(other).faces,
        )


class ReclaimForkTests(TestCase):
    """
    Quiescent pool reclamation (callback before parser disconnect) versus
    callback failure / non-persistence / Connection: close abandonment.
    """

    def setUp(self):
        self.clock = Clock()
        self.ledger = QuietReclaimLedger.declare(self.clock)
        self.addCleanup(self.ledger.retract)

    def test_persistentResponseReclaimsBeforeParserDisconnectAndPools(self):
        pool = HTTPConnectionPool(self.clock, persistent=True)
        endpoint = CapturingEndpoint(pool)
        key = (b"http", b"example", 80)

        protocol, transport, result = completeResponse(pool, endpoint, key)

        self.assertEqual(protocol._state, "QUIESCENT")
        self.assertFalse(transport.disconnecting)
        self.assertIn(protocol, pool._connections[key])

        faces = self.ledger.faces(connection=protocol)
        self.assertEqual(
            faces,
            (
                _reclaim.POOL_PUT,
                _reclaim.RESPONSE_RECLAIM,
                _reclaim.PARSER_DISCONNECT,
            ),
        )
        disconnect = self.ledger.records(
            face=_reclaim.PARSER_DISCONNECT, connection=protocol
        )[0]
        self.assertEqual(disconnect.reason, "ConnectionDone")
        state = self.ledger.stateOf(protocol)
        self.assertTrue(state.reclaimed)
        self.assertEqual(state.poolKeys, frozenset({key}))
        # The idle timeout committed against the cached connection.
        self.assertEqual(len(self.clock.getDelayedCalls()), 1)

    def test_quiescentCallbackFailureLosesConnectionAndDoesNotPool(self):
        def boom(protocol):
            raise RuntimeError("quiescent callback failed")

        protocol, transport = makeProtocol(self.ledger, boom)
        result = protocol.request(FakeRequest())
        responses = []
        result.addCallback(responses.append)
        protocol.dataReceived(EMPTY_200)
        self.flushLoggedErrors(RuntimeError)

        self.assertTrue(transport.disconnecting)
        self.assertEqual(
            self.ledger.faces(connection=protocol),
            (
                _reclaim.RESPONSE_RECLAIM_FAILED,
                _reclaim.PARSER_DISCONNECT,
            ),
        )
        self.assertFalse(self.ledger.stateOf(protocol).reclaimed)

    def test_connectionCloseResponseIsAbandoned(self):
        protocol, transport = makeProtocol(self.ledger)
        result = protocol.request(FakeRequest())
        responses = []
        result.addCallback(responses.append)
        protocol.dataReceived(CLOSING_200)

        # The response itself still succeeds; the connection is abandoned
        # afterwards.
        self.assertEqual(len(responses), 1)
        self.assertEqual(responses[0].code, 200)
        self.assertTrue(transport.disconnecting)
        records = self.ledger.records(connection=protocol)
        self.assertEqual(
            [r.face for r in records],
            [_reclaim.RESPONSE_ABANDON, _reclaim.PARSER_DISCONNECT],
        )
        abandon = records[0]
        self.assertEqual(abandon.reason, "connection-close-header")
        self.assertTrue(self.ledger.stateOf(protocol).abandoned)

    def test_nonPersistentRequestIsAbandoned(self):
        protocol, transport = makeProtocol(self.ledger)
        result = protocol.request(FakeRequest(persistent=False))
        responses = []
        result.addCallback(responses.append)
        protocol.dataReceived(EMPTY_200)

        self.assertEqual(len(responses), 1)
        self.assertTrue(transport.disconnecting)
        abandon = self.ledger.records(
            face=_reclaim.RESPONSE_ABANDON, connection=protocol
        )[0]
        self.assertEqual(abandon.reason, "non-persistent-request")
        self.assertFalse(self.ledger.stateOf(protocol).reclaimed)


class PoolReclaimTests(TestCase):
    """
    Pool put / lease / over-limit eviction / idle loseConnection / batch
    abort-and-join are distinct completion faces on the same reactor.
    """

    def setUp(self):
        self.clock = Clock()
        self.ledger = QuietReclaimLedger.declare(self.clock)
        self.addCleanup(self.ledger.retract)
        self.pool = HTTPConnectionPool(self.clock, persistent=True)
        self.endpoint = CapturingEndpoint(self.pool)
        self.key = (b"http", b"example", 80)

    def test_newConnectionAndPutSettledWithKeyAndGeneration(self):
        protocol, transport, result = completeResponse(
            self.pool, self.endpoint, self.key
        )
        connect = self.ledger.records(face=_reclaim.POOL_CONNECT)
        self.assertEqual(len(connect), 1)
        self.assertEqual(connect[0].key, self.key)
        put = self.ledger.records(face=_reclaim.POOL_PUT)
        self.assertEqual(len(put), 1)
        self.assertIs(put[0].connection, protocol)
        self.assertEqual(put[0].generation, 1)

    def test_cachedCheckoutCancelsIdleTimeoutAndLeases(self):
        first, transport, result = completeResponse(self.pool, self.endpoint, self.key)
        self.assertEqual(len(self.clock.getDelayedCalls()), 1)

        checkout = resultOf(self.pool.getConnection(self.key, self.endpoint))
        self.assertIs(rawProtocol(checkout), first)

        lease = self.ledger.records(face=_reclaim.POOL_LEASE)
        self.assertEqual(len(lease), 1)
        self.assertIs(lease[0].connection, first)
        self.assertTrue(lease[0].detail["timeoutCancelled"])
        # The idle timeout is gone and must never fire for this lease.
        self.assertEqual(self.clock.getDelayedCalls(), [])
        self.clock.advance(1000)
        self.assertFalse(self.ledger.stateOf(first).idleClosed)

    def test_nonQuiescentPutIsRejected(self):
        protocol, transport = makeProtocol(self.ledger)
        self.assertEqual(protocol._state, "QUIESCENT")
        protocol._state = "WAITING"
        self.pool._putConnection(self.key, protocol)
        self.flushLoggedErrors(RuntimeError)

        self.assertNotIn(self.key, self.pool._connections)
        rejected = self.ledger.records(face=_reclaim.POOL_REJECT)
        self.assertEqual(len(rejected), 1)
        self.assertIs(rejected[0].connection, protocol)
        self.assertEqual(rejected[0].detail["state"], "WAITING")
        self.assertEqual(
            self.ledger.records(face=_reclaim.POOL_PUT), ()
        )

    def test_overLimitEvictionLosesHeadBeforePuttingNew(self):
        # Two concurrent connections for the same key ...
        firstCheckout = resultOf(self.pool.getConnection(self.key, self.endpoint))
        secondCheckout = resultOf(self.pool.getConnection(self.key, self.endpoint))
        first = rawProtocol(firstCheckout)
        second = rawProtocol(secondCheckout)
        self.assertIsNot(first, second)

        # ... both complete and enter the cache.
        firstResult = firstCheckout.request(FakeRequest())
        first.dataReceived(EMPTY_200)
        secondResult = secondCheckout.request(FakeRequest())
        second.dataReceived(EMPTY_200)

        # A third fresh connection completes while the cache is full.
        thirdCheckout = resultOf(self.pool._newConnection(self.key, self.endpoint))
        third = rawProtocol(thirdCheckout)
        thirdResult = thirdCheckout.request(FakeRequest())
        third.dataReceived(EMPTY_200)

        evictions = self.ledger.records(face=_reclaim.POOL_EVICT)
        self.assertEqual(len(evictions), 1)
        eviction = evictions[0]
        self.assertIs(eviction.connection, first)
        self.assertEqual(eviction.reason, "maxPersistentPerHost")
        self.assertTrue(first.transport.disconnecting)
        self.assertFalse(first.transport.disconnected)

        # Eviction settled strictly before the new pool put.
        putThird = [
            record
            for record in self.ledger.records(face=_reclaim.POOL_PUT)
            if record.connection is third
        ][0]
        self.assertLess(eviction.seq, putThird.seq)

        # Pool membership and timeout handles match the settlement.
        self.assertEqual(self.pool._connections[self.key], [second, third])
        self.assertNotIn(first, self.pool._timeouts)
        self.assertEqual(
            {self.pool._timeouts[second], self.pool._timeouts[third]},
            set(self.clock.getDelayedCalls()),
        )
        self.assertTrue(self.ledger.stateOf(first).evicted)

    def test_idleTimeoutLosesConnectionOnItsOwnFace(self):
        protocol, transport, result = completeResponse(
            self.pool, self.endpoint, self.key
        )
        self.clock.advance(self.pool.cachedConnectionTimeout)

        self.assertTrue(transport.disconnecting)
        self.assertEqual(self.pool._connections[self.key], [])
        self.assertEqual(
            self.ledger.records(face=_reclaim.POOL_IDLE, connection=protocol).__len__(),
            1,
        )
        self.assertTrue(self.ledger.stateOf(protocol).idleClosed)
        # Idle timeout is neither eviction nor batch-abort.
        self.assertEqual(self.ledger.records(face=_reclaim.POOL_EVICT), ())
        self.assertEqual(self.ledger.records(face=_reclaim.POOL_CLOSE_ABORT), ())

    def test_closeCachedConnectionsJoinsOnlyAfterEveryAbortCompleted(self):
        # Two cached connections.
        firstCheckout = resultOf(self.pool.getConnection(self.key, self.endpoint))
        secondCheckout = resultOf(self.pool.getConnection(self.key, self.endpoint))
        first = rawProtocol(firstCheckout)
        second = rawProtocol(secondCheckout)
        firstCheckout.request(FakeRequest())
        first.dataReceived(EMPTY_200)
        secondCheckout.request(FakeRequest())
        second.dataReceived(EMPTY_200)

        closed = []
        joinDeferred = self.pool.closeCachedConnections()
        joinDeferred.addCallback(closed.append)

        # Every cached connection got a batch-abort face and was asked to
        # close (loseConnection via protocol.abort), but the join must not
        # complete yet.
        self.assertEqual(
            {
                r.connection
                for r in self.ledger.records(face=_reclaim.POOL_CLOSE_ABORT)
            },
            {first, second},
        )
        self.assertEqual(closed, [])
        self.assertEqual(self.ledger.records(face=_reclaim.POOL_CLOSE_JOIN), ())
        for protocol in (first, second):
            self.assertEqual(protocol._state, "ABORTING")
            self.assertTrue(protocol.transport.disconnecting)
            # Batch close asks for a graceful lose, not abortConnection.
            self.assertFalse(protocol.transport.disconnected)
        # The idle timeout handles were cancelled.
        self.assertEqual(self.clock.getDelayedCalls(), [])

        # Completing only one abort still does not join.
        first.connectionLost(Failure(ConnectionDone("first gone")))
        self.assertEqual(closed, [])
        self.assertEqual(self.ledger.records(face=_reclaim.POOL_CLOSE_JOIN), ())

        second.connectionLost(Failure(ConnectionDone("second gone")))
        self.assertEqual(closed, [None])
        join = self.ledger.records(face=_reclaim.POOL_CLOSE_JOIN)
        self.assertEqual(len(join), 1)
        self.assertEqual(join[0].detail["closed"], 2)
        for protocol in (first, second):
            self.assertTrue(self.ledger.stateOf(protocol).batchClosed)
        self.assertEqual(self.pool._connections, {})
        self.assertEqual(self.pool._timeouts, {})


@implementer(IAgentEndpointFactory)
class RecordingEndpointFactory:
    """
    Endpoint factory handing out endpoints that build real protocols and
    capture them.
    """

    def __init__(self):
        self.bucket = []

    def endpointForURI(self, uri):
        return _BucketEndpoint(self.bucket)


@implementer(IStreamClientEndpoint)
class _BucketEndpoint:
    def __init__(self, bucket):
        self._bucket = bucket

    def connect(self, factory):
        protocol = factory.buildProtocol(None)
        transport = StringTransport()
        protocol.makeConnection(transport)
        self._bucket.append((protocol, transport))
        return succeed(protocol)


@implementer(IStreamClientEndpoint)
class _ProxyBucketEndpoint:
    """
    A single endpoint object reused for all proxied requests.
    """

    def __init__(self):
        self.bucket = []

    def connect(self, factory):
        protocol = factory.buildProtocol(None)
        transport = StringTransport()
        protocol.makeConnection(transport)
        self.bucket.append((protocol, transport))
        return succeed(protocol)


class PoolKeyDimensionTests(TestCase):
    """
    Direct Agent pools per origin key versus ProxyAgent's single proxy key.
    """

    def setUp(self):
        self.clock = Clock()
        self.ledger = QuietReclaimLedger.declare(self.clock)
        self.addCleanup(self.ledger.retract)

    def _completeAgentRequest(self, agent, bucket, url):
        result = agent.request(b"GET", url)
        responses = []
        result.addCallback(responses.append)
        protocol, transport = bucket[-1]
        protocol.dataReceived(EMPTY_200)
        return protocol, responses[0]

    def test_agentSettlesUnderDistinctOriginKeys(self):
        factory = RecordingEndpointFactory()
        pool = HTTPConnectionPool(self.clock, persistent=True)
        agent = Agent.usingEndpointFactory(
            self.clock, factory, pool=pool
        )

        p1, response1 = self._completeAgentRequest(
            agent, factory.bucket, b"http://a.example/"
        )
        p2, response2 = self._completeAgentRequest(
            agent, factory.bucket, b"http://b.example/"
        )
        self.assertIsNot(p1, p2)

        puts = self.ledger.records(face=_reclaim.POOL_PUT)
        keys = {record.key for record in puts}
        self.assertEqual(
            keys,
            {(b"http", b"a.example", 80), (b"http", b"b.example", 80)},
        )
        keyOfP1 = [r.key for r in puts if r.connection is p1][0]
        keyOfP2 = [r.key for r in puts if r.connection is p2][0]
        self.assertNotEqual(keyOfP1, keyOfP2)

    def test_proxyAgentSettlesEverythingUnderOneProxyKey(self):
        proxyEndpoint = _ProxyBucketEndpoint()
        pool = HTTPConnectionPool(self.clock, persistent=True)
        agent = ProxyAgent(proxyEndpoint, reactor=self.clock, pool=pool)

        # Two requests for DIFFERENT origins, issued concurrently, use two
        # distinct connections to the proxy ...
        result1 = agent.request(b"GET", b"http://a.example/")
        result2 = agent.request(b"GET", b"http://b.example/")
        responses = []
        result1.addCallback(responses.append)
        result2.addCallback(responses.append)
        p1 = proxyEndpoint.bucket[0][0]
        p2 = proxyEndpoint.bucket[1][0]
        self.assertIsNot(p1, p2)
        p1.dataReceived(EMPTY_200)
        p2.dataReceived(EMPTY_200)
        self.assertEqual(len(responses), 2)

        # ... but every settlement - connection, quiet reclaim, membership -
        # happens under the single proxy key, not per origin.
        connectKeys = {
            record.key for record in self.ledger.records(face=_reclaim.POOL_CONNECT)
        }
        putKeys = {
            record.key for record in self.ledger.records(face=_reclaim.POOL_PUT)
        }
        expectedKey = ("http-proxy", proxyEndpoint)
        self.assertEqual(connectKeys, {expectedKey})
        self.assertEqual(putKeys, {expectedKey})
        self.assertEqual(self.ledger.poolKeys(), frozenset({expectedKey}))
        self.assertEqual(
            {r.connection for r in self.ledger.records(face=_reclaim.POOL_PUT)},
            {p1, p2},
        )


class SameReactorInterleavingTests(TestCase):
    """
    Nested/interleaved work on ONE reactor: an outer request holding a lease
    and reading a body, while an inner layer performs a nested request, a
    readBody cancel, an idle timeout and a full reclaim on other keys.
    """

    def setUp(self):
        self.clock = Clock()
        self.ledger = QuietReclaimLedger.declare(self.clock)
        self.addCleanup(self.ledger.retract)
        self.pool = HTTPConnectionPool(self.clock, persistent=True)
        # The outer layer's pooled connection gets a longer idle timeout so
        # that the inner layer's clock advance cannot touch it: this proves
        # frame exit, rather than timeout coincidence, preserves the outer
        # lease.
        self.poolC = HTTPConnectionPool(self.clock, persistent=True)
        self.poolC.cachedConnectionTimeout = 500
        self.keyA = (b"http", b"a.example", 80)
        self.keyB = (b"http", b"b.example", 80)
        self.keyC = (b"http", b"c.example", 80)
        self.endpointA = CapturingEndpoint(self.pool)
        self.endpointB = CapturingEndpoint(self.pool)
        self.endpointC = CapturingEndpoint(self.poolC)

    def test_innerLayerCannotWriteThroughOuterAndEffectsDoNotRevoke(self):
        # --- Outer layer: check out / own one busy connection and quietly
        # reclaim another one. ---------------------------------------------
        with self.ledger.layer() as outer:
            busyCheckout = resultOf(
                self.pool.getConnection(self.keyA, self.endpointA)
            )
            busy = rawProtocol(busyCheckout)
            busyResult = busyCheckout.request(FakeRequest())
            busyResponses = []
            busyResult.addCallback(busyResponses.append)
            busy.dataReceived(PARTIAL_200)
            busyResponse = busyResponses[0]

            cachedCheckout = resultOf(
                self.poolC.getConnection(self.keyC, self.endpointC)
            )
            cached = rawProtocol(cachedCheckout)
            cachedResult = cachedCheckout.request(FakeRequest())
            cached.dataReceived(EMPTY_200)
            self.assertIn(cached, self.poolC._connections[self.keyC])
            cachedPut = [
                r
                for r in self.ledger.records(face=_reclaim.POOL_PUT)
                if r.connection is cached
            ][0]
            self.assertEqual(cachedPut.frame, outer.path)

            # --- Inner layer interleaves on the same reactor. -------------
            with self.ledger.layer() as inner:
                # A nested request on the outer busy connection is refused:
                # the outer generation is not written through.
                nested = busy.request(FakeRequest())
                failureOf(nested).trap(RequestNotSent)
                self.assertEqual(self.ledger.generations(busy), (1,))

                # The outer body is cancelled from inside: direct abort,
                # settled to the inner frame against the outer generation.
                body = readBody(busyResponse)
                body.cancel()
                failureOf(body).trap(CancelledError)

                # Another key is fully reclaimed and idle-times-out inside
                # the inner layer.
                shortCheckout = resultOf(
                    self.pool.getConnection(self.keyB, self.endpointB)
                )
                short = rawProtocol(shortCheckout)
                shortResult = shortCheckout.request(FakeRequest())
                short.dataReceived(EMPTY_200)
                self.clock.advance(self.pool.cachedConnectionTimeout)

                innerFaces = self.ledger.faces(frame=inner, deep=False)
                self.assertEqual(
                    innerFaces,
                    (
                        _reclaim.REQUEST_NOT_SENT,
                        _reclaim.READ_BODY_CANCEL_ABORT,
                        _reclaim.POOL_CONNECT,
                        _reclaim.POOL_PUT,
                        _reclaim.RESPONSE_RECLAIM,
                        _reclaim.PARSER_DISCONNECT,
                        _reclaim.POOL_IDLE,
                    ),
                )
                # The readBody cancel never entered pool accounting.
                self.assertNotIn(
                    _reclaim.POOL_LEASE,
                    innerFaces,
                )

            # Inner layer committed: the frame is frozen ...
            self.assertTrue(inner.committed)
            self.assertRaises(
                RuntimeError, inner.settle, _reclaim.POOL_PUT
            )
            # ... and nothing about the outer layer's direct settlements
            # changed: only the two connects and the C reclaim happened
            # directly in the outer frame.
            self.assertEqual(
                self.ledger.faces(frame=outer, deep=False),
                (
                    _reclaim.POOL_CONNECT,
                    _reclaim.POOL_CONNECT,
                    _reclaim.POOL_PUT,
                    _reclaim.RESPONSE_RECLAIM,
                    _reclaim.PARSER_DISCONNECT,
                ),
            )

        # --- Both layers gone. Side effects are not revoked: -------------
        # The readBody abort of the outer connection stands.
        self.assertTrue(busy.transport.disconnected)
        self.assertTrue(self.ledger.stateOf(busy).aborted)
        busyAbort = self.ledger.records(
            face=_reclaim.READ_BODY_CANCEL_ABORT, connection=busy
        )
        self.assertEqual(busyAbort[0].frame, inner.path)
        self.assertEqual(busyAbort[0].generation, 1)

        # The idle timeout of the inner connection stands and is not
        # resurrected into the cache.
        self.assertTrue(self.ledger.stateOf(short).idleClosed)
        self.assertEqual(self.pool._connections[self.keyB], [])

        # The outer layer's quietly pooled connection is still pooled, with
        # its committed idle-timeout handle still live; frame exit did not
        # revoke the membership.
        self.assertEqual(self.poolC._connections[self.keyC], [cached])
        cachedTimeout = self.poolC._timeouts[cached]
        self.assertFalse(cachedTimeout.cancelled)
        self.assertIn(cachedTimeout, self.clock.getDelayedCalls())

        # That committed timeout (500s, the clock is already at 240s) later
        # fires into the root layer.
        self.clock.advance(self.poolC.cachedConnectionTimeout)
        cachedIdle = self.ledger.records(
            face=_reclaim.POOL_IDLE, connection=cached
        )
        self.assertEqual(len(cachedIdle), 1)
        self.assertEqual(cachedIdle[0].frame, ())

    def test_batchCloseInsideInnerLayerStaysSettledAfterExit(self):
        with self.ledger.layer() as outer:
            checkout = resultOf(self.pool.getConnection(self.keyA, self.endpointA))
            cached = rawProtocol(checkout)
            checkout.request(FakeRequest())
            cached.dataReceived(EMPTY_200)

            with self.ledger.layer() as inner:
                joined = []
                closer = self.pool.closeCachedConnections()
                closer.addCallback(joined.append)
                self.assertEqual(joined, [])
                cached.connectionLost(Failure(ConnectionDone("gone")))
                self.assertEqual(joined, [None])
                self.assertEqual(
                    self.ledger.faces(frame=inner, deep=False),
                    (
                        _reclaim.POOL_CLOSE_ABORT,
                        _reclaim.PROTOCOL_ABORT,
                        _reclaim.POOL_CLOSE_JOIN,
                    ),
                )

        # After both layers exit the batch-abort cannot be undone.
        self.assertTrue(self.ledger.stateOf(cached).batchClosed)
        self.assertEqual(self.pool._connections, {})
        self.assertEqual(
            len(self.ledger.records(face=_reclaim.POOL_CLOSE_JOIN)), 1
        )


class PinCompatibilityTests(TestCase):
    """
    Without a declared ledger every exit keeps its pinned behaviour.
    """

    def setUp(self):
        self.clock = Clock()

    def test_twoArgFactoryStillAccepted(self):
        """
        Test doubles replacing C{pool._factory} with the historical
        two-argument constructor keep working.
        """

        class StubProtocol:
            state = "QUIESCENT"
            transport = StringTransport()
            _requestGeneration = 0

        class DummyFactory(protocol.Factory):
            protocol = StubProtocol

            def __init__(self, quiescentCallback, metadata):
                self.quiescentCallback = quiescentCallback
                self.metadata = metadata

        pool = HTTPConnectionPool(self.clock)
        pool._factory = DummyFactory

        @implementer(IStreamClientEndpoint)
        class Endpoint:
            def connect(self, factory):
                return succeed(factory.buildProtocol(None))

        connection = resultOf(pool.getConnection((b"http", b"x", 80), Endpoint()))
        self.assertIsInstance(rawProtocol(connection), StubProtocol)

    def test_abortCancelBehavesAsPinnedWithoutLedger(self):
        client, transport = makeProtocol(None)
        result = client.request(FakeRequest())
        result.cancel()
        failure = failureOf(result)
        failure.trap(ResponseNeverReceived)
        self.assertTrue(transport.disconnected)

    def test_quiescentPutBehavesAsPinnedWithoutLedger(self):
        pool = HTTPConnectionPool(self.clock, persistent=True)
        endpoint = CapturingEndpoint(pool)
        key = (b"http", b"example", 80)
        client, transport, result = completeResponse(pool, endpoint, key)
        self.assertIn(client, pool._connections[key])
        self.assertFalse(transport.disconnecting)

    def test_responseWithoutTokenKeepsDeprecatedAbortBehaviour(self):
        """
        Responses produced without a ledger still warn and do nothing when a
        body is cancelled on a transport lacking abortConnection.
        """
        import warnings

        class NonAbortingTransport(StringTransport):
            abortConnection = None

        bodyTransport = NonAbortingTransport()
        response = Response._construct(
            (b"HTTP", 1, 1),
            200,
            b"OK",
            Headers(),
            bodyTransport,
            FakeRequest(),
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            body = readBody(response)
        self.assertTrue(
            any(issubclass(w.category, DeprecationWarning) for w in caught)
        )
        # Cancelling does nothing because there is no abort and no token.
        body.cancel()
        failureOf(body).trap(Exception)
        self.assertFalse(bodyTransport.disconnected)
