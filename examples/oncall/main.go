// Command oncall opens a terminal chat with an on-call engineer for a
// fictional shop. It checks services, sends an investigator subagent to dig
// through logs and metrics, and asks before it restarts or rolls back anything.
//
//	go run ./examples/oncall
//
// Try: "Checkout is slow for customers, what's going on?"
package main

import (
	"context"
	"fmt"
	"log"
	"strings"
	"time"

	"crux.foo"
)

// All services, deploys, logs and metrics below are fictional fixtures.

type Service struct {
	Name       string `json:"name"`
	Status     string `json:"status"`
	Version    string `json:"version"`
	DeployedAt string `json:"deployed_at"`
	Previous   string `json:"previous_version"`
}

var services = map[string]Service{
	"checkout": {"checkout", "degraded", "v3.8.1", "2026-09-30T13:40Z", "v3.8.0"},
	"payments": {"payments", "degraded", "v2.14.0", "2026-09-30T14:02Z", "v2.13.4"},
	"search":   {"search", "healthy", "v1.22.3", "2026-09-28T09:15Z", "v1.22.2"},
	"catalog":  {"catalog", "healthy", "v5.1.0", "2026-09-25T17:30Z", "v5.0.9"},
}

var logs = map[string][]string{
	"checkout": {
		"14:03:12 WARN  upstream payments slow: 4812ms (budget 800ms)",
		"14:03:40 ERROR order 88213 failed: payments timeout after 5000ms",
		"14:04:02 WARN  retry queue at 1,204 orders",
	},
	"payments": {
		"14:02:07 INFO  starting payments v2.14.0 (pool size 10, was 50)",
		"14:02:51 WARN  db pool exhausted, 37 requests waiting",
		"14:03:18 ERROR acquire connection: context deadline exceeded",
		"14:04:30 WARN  db pool exhausted, 112 requests waiting",
	},
	"search":  {"14:00:00 INFO  reindex finished in 41s"},
	"catalog": {"13:58:12 INFO  cache warm, hit rate 97%"},
}

var metrics = map[string]map[string]string{
	"checkout": {"p99_latency_ms": "5210 (was 310 before 14:02)", "error_rate": "8.4% (was 0.1%)", "rps": "220"},
	"payments": {"p99_latency_ms": "4930 (was 180 before 14:02)", "error_rate": "11.2% (was 0.0%)", "db_pool_in_use": "10/10"},
	"search":   {"p99_latency_ms": "95", "error_rate": "0.0%", "rps": "610"},
	"catalog":  {"p99_latency_ms": "40", "error_rate": "0.0%", "rps": "1400"},
}

type serviceArgs struct {
	Service string `json:"service" description:"Service name: checkout, payments, search or catalog"`
}

type logArgs struct {
	Service string `json:"service" description:"Service name"`
	Level   string `json:"level,omitempty" description:"Only lines at this level or worse: INFO, WARN or ERROR"`
}

type metricArgs struct {
	Service string `json:"service" description:"Service name"`
	Metric  string `json:"metric" description:"p99_latency_ms, error_rate, rps or db_pool_in_use"`
}

type rollbackArgs struct {
	Service string `json:"service" description:"Service name"`
	Version string `json:"version" description:"Version to roll back to"`
}

func lookup(name string) (Service, error) {
	svc, ok := services[strings.ToLower(name)]
	if !ok {
		return Service{}, fmt.Errorf("unknown service %q", name)
	}
	return svc, nil
}

// observability groups the read-only tools, so agents can take them all with
// WithToolsets("observability") and the chat's sidebar groups them.
type observability struct{}

func (observability) Register(reg crux.ToolsRegistry) error {
	set := crux.WithToolset("observability")

	crux.RegisterToolWithRegistry(reg, "list_services", "List every service with its status and current version",
		func(ctx context.Context, _ struct{}) ([]Service, *crux.StateDelta, error) {
			time.Sleep(300 * time.Millisecond)
			var out []Service
			for _, name := range []string{"checkout", "payments", "search", "catalog"} {
				out = append(out, services[name])
			}
			return out, nil, nil
		}, set)

	crux.RegisterToolWithRegistry(reg, "query_logs", "Read a service's recent log lines, optionally only at a level or worse",
		func(ctx context.Context, args logArgs) ([]string, *crux.StateDelta, error) {
			time.Sleep(1200 * time.Millisecond) // log search is slow
			if _, err := lookup(args.Service); err != nil {
				return nil, nil, err
			}
			rank := map[string]int{"INFO": 0, "WARN": 1, "ERROR": 2}
			least := rank[strings.ToUpper(args.Level)]
			var out []string
			for _, line := range logs[strings.ToLower(args.Service)] {
				for level, r := range rank {
					if r >= least && strings.Contains(line, " "+level+" ") {
						out = append(out, line)
					}
				}
			}
			return out, nil, nil
		}, set)

	crux.RegisterToolWithRegistry(reg, "get_metric", "Read one metric of a service over the last 30 minutes",
		func(ctx context.Context, args metricArgs) (string, *crux.StateDelta, error) {
			time.Sleep(600 * time.Millisecond)
			if _, err := lookup(args.Service); err != nil {
				return "", nil, err
			}
			value, ok := metrics[strings.ToLower(args.Service)][args.Metric]
			if !ok {
				return "", nil, fmt.Errorf("%s has no metric %q", args.Service, args.Metric)
			}
			return value, nil, nil
		}, set)
	return nil
}

func init() {
	if err := crux.AddToolset(observability{}); err != nil {
		log.Fatal(err)
	}

	crux.RegisterTool("service_status", "Get a service's status, version and last deploy",
		func(ctx context.Context, args serviceArgs) (Service, error) {
			time.Sleep(400 * time.Millisecond)
			return lookup(args.Service)
		})

	// Anything that changes production waits for a person to approve it.
	crux.RegisterTool("restart_service", "Restart every instance of a service",
		func(ctx context.Context, args serviceArgs) (string, error) {
			time.Sleep(2 * time.Second)
			if _, err := lookup(args.Service); err != nil {
				return "", err
			}
			return args.Service + " restarted; 6/6 instances healthy", nil
		}, crux.WithApprovalNeeded(true))

	crux.RegisterTool("rollback_deploy", "Roll a service back to an earlier version",
		func(ctx context.Context, args rollbackArgs) (string, error) {
			time.Sleep(2500 * time.Millisecond)
			if _, err := lookup(args.Service); err != nil {
				return "", err
			}
			return fmt.Sprintf("%s rolled back to %s; error rate back to 0.0%%", args.Service, args.Version), nil
		}, crux.WithApprovalNeeded(true))
}

func main() {
	investigator := crux.Must(crux.New("investigator", crux.ClaudeSonnet5_5,
		crux.WithInstructions(`You investigate production incidents. Read logs and metrics of
the services you are asked about, and of their dependencies, until you find the
most likely root cause. Answer with the cause, the evidence, and the time it started.`),
		crux.WithToolsets("observability"),
	))

	oncall := crux.Must(crux.New("oncall", crux.ClaudeSonnet5_5,
		crux.WithInstructions(`You are the on-call engineer for a web shop. Check the status of
the services involved, send the investigator to find the root cause, then propose
and carry out the smallest fix, such as rolling back a bad deploy. Keep answers short
and use markdown: a one-line summary, then a table of what you found.`),
		crux.WithTools([]string{"service_status", "restart_service", "rollback_deploy"}),
		crux.WithSubAgent(investigator, "Finds the root cause of an incident from logs and metrics"),
		crux.WithReasoning(crux.ReasoningLow),
		crux.WithMaxTurns(15),
	))

	if err := crux.CLI(oncall); err != nil {
		log.Fatal(err)
	}
}
