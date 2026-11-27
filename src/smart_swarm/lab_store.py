"""Durable action correlation for the project facade; native journals remain authoritative."""
from contextlib import contextmanager
from hashlib import sha256
import json
import sqlite3
import time
from uuid import uuid4
from .contract import json_bytes


class ActionStore:
    def __init__(self, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        with self.connection() as db:
            db.execute('CREATE TABLE IF NOT EXISTS actions (id TEXT PRIMARY KEY, key TEXT UNIQUE, digest TEXT, request TEXT, result TEXT)')
            db.execute('CREATE TABLE IF NOT EXISTS events (seq INTEGER PRIMARY KEY, at REAL, body TEXT)')
            db.execute('CREATE TABLE IF NOT EXISTS revisions (task TEXT, version INTEGER, digest TEXT, PRIMARY KEY(task,version))')

    @contextmanager
    def connection(self):
        with sqlite3.connect(self.path, timeout=5) as db:
            db.execute('PRAGMA busy_timeout=5000')
            db.execute('BEGIN IMMEDIATE')
            yield db

    def reserve(self, request, revision):
        digest = sha256(json_bytes(request)).hexdigest()
        with self.connection() as db:
            prior = db.execute('SELECT digest,result FROM actions WHERE key=?', (request['idempotency_key'],)).fetchone()
            if prior:
                if prior[0] != digest:
                    raise ValueError('Idempotency key was already used for another request')
                return json.loads(prior[1]), False
            if request['expected_revision'] != revision:
                raise ValueError('STALE_REVISION. Refresh the project snapshot')
            identifier = uuid4().hex
            result = {'status': 'pending', 'operation_id': identifier, 'resulting_revision': revision,
                      'reason': None, 'native_acknowledgement': None, 'receipts': []}
            db.execute('INSERT INTO actions VALUES (?,?,?,?,?)',
                       (identifier, request['idempotency_key'], digest, json.dumps(request), json.dumps(result)))
            self.event(db, result)
            return result, True

    def existing(self, request):
        digest = sha256(json_bytes(request)).hexdigest()
        with self.connection() as db:
            prior = db.execute('SELECT digest,result FROM actions WHERE key=?', (request['idempotency_key'],)).fetchone()
        if prior and prior[0] != digest:
            raise ValueError('Idempotency key was already used for another request')
        return json.loads(prior[1]) if prior else None

    def event(self, db, value):
        db.execute('INSERT INTO events(at,body) VALUES (?,?)', (time.time(), json.dumps(value)))

    def revision(self, task, digest):
        with self.connection() as db:
            row = db.execute('SELECT version,digest FROM revisions WHERE task=? ORDER BY version DESC LIMIT 1',(task,)).fetchone()
            if row and row[1] == digest:
                return row[0]
            version = row[0]+1 if row else 0
            db.execute('INSERT INTO revisions VALUES (?,?,?)',(task,version,digest))
        return version

    def revision_digest(self, task, version):
        if type(version) is not int or version < 0:
            raise ValueError('Project revision must be a nonnegative integer')
        with self.connection() as db:
            row = db.execute('SELECT digest FROM revisions WHERE task=? AND version=?',(task,version)).fetchone()
        if row is None:
            raise ValueError('Unknown project revision')
        return row[0]

    def number_for_digest(self, task, digest):
        with self.connection() as db:
            row = db.execute('SELECT version FROM revisions WHERE task=? AND digest=? ORDER BY version DESC LIMIT 1',(task,digest)).fetchone()
        return row[0] if row else self.revision(task,digest)

    def request(self, identifier):
        with self.connection() as db:
            row = db.execute('SELECT request FROM actions WHERE id=?',(identifier,)).fetchone()
        if row is None:
            raise ValueError('Unknown project operation')
        return json.loads(row[0])

    def complete(self, identifier, result):
        with self.connection() as db:
            db.execute('UPDATE actions SET result=? WHERE id=?', (json.dumps(result), identifier))
            self.event(db, result)
        return result

    def operation(self, identifier):
        with self.connection() as db:
            row = db.execute('SELECT result FROM actions WHERE id=?', (identifier,)).fetchone()
        if row is None:
            raise ValueError('Unknown operation ID')
        return json.loads(row[0])

    def events(self, after=0, limit=50):
        if type(after) is not int or after < 0 or type(limit) is not int or not 1 <= limit <= 100:
            raise ValueError('Event cursor must be nonnegative and limit must be 1..100')
        with self.connection() as db:
            found = db.execute('SELECT seq,at,body FROM events WHERE seq>? ORDER BY seq LIMIT ?', (after,limit)).fetchall()
        return [{'sequence': seq, 'created_at': at, **json.loads(body)} for seq, at, body in found]

    def interrupted(self, statuses=('pending',)):
        with self.connection() as db:
            rows = db.execute('SELECT id,request,result FROM actions').fetchall()
        return [(key, json.loads(request)) for key, request, raw in rows if json.loads(raw)['status'] in statuses]
