package crux

import (
	"context"
	"fmt"
	"maps"
	"os"
	"path/filepath"
	"slices"

	"github.com/apzuk3/crux/internal/filesystem"
	"github.com/apzuk3/crux/internal/network"
)

// The built-in toolsets: Filesystem and Network. Their tools are implemented
// in internal packages; this file holds their public names and options and
// registers them.

// registerTextTool registers a tool whose output is text and that changes no
// session state.
func registerTextTool[In any](registry ToolsRegistry, name, description string, fn func(context.Context, In) (string, error), opts ...ToolOption) {
	RegisterToolWithRegistry(registry, name, description, func(ctx context.Context, in In) (string, *StateDelta, error) {
		out, err := fn(ctx, in)
		return out, nil, err
	}, opts...)
}

// ---------- filesystem ----------

// Filesystem tool names.
const (
	FsReadFile           = "read_file"
	FsReadMultipleFiles  = "read_multiple_files"
	FsListDirectory      = "list_directory"
	FsDirectoryTree      = "directory_tree"
	FsGlob               = "glob"
	FsSearchFilesContent = "search_files_content"
	FsWriteFile          = "write_file"
	FsEditFile           = "edit_file"
	FsCreateDirectory    = "create_directory"
	FsRemoveDirectory    = "remove_directory"
)

// ToolsetFilesystem is the name of the toolset registered by Filesystem.
const ToolsetFilesystem = "filesystem"

// Filesystem returns a toolset of file tools confined to root. Paths the model
// passes are relative to root (absolute paths are accepted when they are
// inside it), and nothing outside root can be reached, including through
// ".." or symlinks. Paths use forward slashes on every platform.
//
// Read-only tools: read_file, read_multiple_files, list_directory,
// directory_tree, glob, search_files_content.
// Tools that change files need approval: write_file, edit_file,
// create_directory, remove_directory.
//
// Read-only tools run without approval, so the model can read any file under
// root, including secrets such as .env files or private keys, and its content
// is sent to the provider. Scope root narrowly to the files the agent needs.
//
// Calls from one model turn run one at a time, in the order the model wrote
// them, so creating a directory and then writing a file into it works.
//
// Output and work per call are bounded: large files, long listings and
// searches are cut off with a note telling the model how to narrow the
// request.
//
// An AGENTS.md at root is added to the instructions of every request from an
// agent with any of these tools, read again before each request, so the
// agent follows the directory's conventions as coding agents do. Turn this
// off with WithFilesystemAgentsFile(false).
//
// The tools belong to the "filesystem" toolset, so WithToolsets("filesystem")
// gives an agent all of them. Register fails when root is not an existing
// directory.
func Filesystem(root string, opts ...FilesystemOption) Toolset {
	f := &filesystemToolset{root: root, agentsFile: true}
	for _, opt := range opts {
		opt(f)
	}
	return f
}

// FilesystemOption configures the Filesystem toolset.
type FilesystemOption func(*filesystemToolset)

// WithFilesystemAgentsFile sets whether the AGENTS.md at the root is added to
// the agent's instructions. The default is true.
func WithFilesystemAgentsFile(enabled bool) FilesystemOption {
	return func(f *filesystemToolset) { f.agentsFile = enabled }
}

type filesystemToolset struct {
	root       string
	agentsFile bool
}

func (f *filesystemToolset) Register(registry ToolsRegistry) error {
	root, err := filepath.Abs(f.root)
	if err != nil {
		return fmt.Errorf("filesystem root %q: %w", f.root, err)
	}
	info, err := os.Stat(root)
	if err != nil {
		return fmt.Errorf("filesystem root: %w", err)
	}
	if !info.IsDir() {
		return fmt.Errorf("filesystem root %q is not a directory", f.root)
	}

	t := filesystem.New(root)
	approval := WithApprovalNeeded(true)
	// Every file tool is sequential, so calls from one turn see each other's
	// changes in the order the model wrote them.
	inToolset, sequential := WithToolset(ToolsetFilesystem), WithSequential()
	inSet := func(tool *Tool) { inToolset(tool); sequential(tool) }
	if f.agentsFile {
		// One option for every tool, so AGENTS.md is added once per request.
		inSequentialSet, agents := inSet, withInstructions(t.AgentsInstructions)
		inSet = func(tool *Tool) { inSequentialSet(tool); agents(tool) }
	}

	registerTextTool(registry, FsReadFile, "Read a text file. The whole file is returned unless line (1-based start line) and limit (maximum number of lines) select a range.", t.ReadFile, inSet)
	registerTextTool(registry, FsReadMultipleFiles, fmt.Sprintf("Read several text files at once (at most %d). Prefer this over sequential read_file calls.", filesystem.MaxMultiReadFiles), t.ReadMultipleFiles, inSet)
	registerTextTool(registry, FsListDirectory, "List the files and directories directly inside a directory.", t.ListDirectory, inSet)
	registerTextTool(registry, FsDirectoryTree, "Show a recursive tree of files and directories.", t.DirectoryTree, inSet)
	registerTextTool(registry, FsGlob, "Find files whose path matches a glob pattern such as **/*.go or src/*.ts. ** matches any number of directories.", t.Glob, inSet)
	registerTextTool(registry, FsSearchFilesContent, "Search file contents for text or a regular expression. Returns matches as path:line:column: text.", t.SearchFilesContent, inSet)
	registerTextTool(registry, FsWriteFile, "Create a file, or completely overwrite an existing one. Missing parent directories are created.", t.WriteFile, inSet, approval)
	registerTextTool(registry, FsEditFile, "Edit a text file by replacing exact text. Each old_text must appear exactly once in the file; include enough surrounding text to make it unique. Edits are applied in order and either all succeed or none are written. In a file with CRLF line endings, an old_text that is only found once its \\n line endings become \\r\\n is matched that way, and the \\n line endings in its new_text are then written as \\r\\n too.", t.EditFile, inSet, approval)
	registerTextTool(registry, FsCreateDirectory, "Create one or more directories, including missing parents.", t.CreateDirectory, inSet, approval)
	registerTextTool(registry, FsRemoveDirectory, "Remove one or more empty directories.", t.RemoveDirectory, inSet, approval)

	return nil
}

// ---------- network ----------

// Network tool names.
const (
	NetDNSLookup   = "dns_lookup"
	NetWhois       = "whois"
	NetHTTPGet     = "http_get"
	NetHTTPRequest = "http_request"
	NetConnect     = "net_connect"
	NetSend        = "net_send"
	NetRead        = "net_read"
	NetList        = "net_list"
	NetClose       = "net_close"
	NetListen      = "net_listen"
	NetHTTPServe   = "http_serve"
)

// ToolsetNetwork is the name of the toolset registered by Network.
const ToolsetNetwork = "network"

// netApprovalDefaults says which tools need approval: those that send data
// the model chose or open a port. http_get only reaches public addresses.
var netApprovalDefaults = map[string]bool{
	NetDNSLookup:   false,
	NetWhois:       false,
	NetHTTPGet:     false,
	NetHTTPRequest: true,
	NetConnect:     true,
	NetSend:        true,
	NetRead:        false,
	NetList:        false,
	NetClose:       false,
	NetListen:      true,
	NetHTTPServe:   true,
}

// netSequential lists the tools that act on an open handle. Their calls from
// one turn run in the model's order, so a read follows the send before it.
var netSequential = map[string]bool{NetSend: true, NetRead: true, NetClose: true}

// NetworkOption configures the Network toolset.
type NetworkOption func(*networkConfig)

type networkConfig struct {
	private  bool
	hosts    []string
	approval []approvalRule
}

// WithNetworkPrivate sets whether the tools may reach loopback, private,
// link-local (including cloud metadata at 169.254.169.254), CGNAT and
// unique-local addresses, and unix sockets. The default is true. The check
// is made on the address actually dialed, after DNS and on every redirect.
// http_get never reaches them, whatever this says.
func WithNetworkPrivate(allowed bool) NetworkOption {
	return func(c *networkConfig) { c.private = allowed }
}

// WithNetworkHosts limits the tools to these hosts: exact names, wildcards
// such as "*.example.com", IP addresses or CIDR ranges such as
// "10.0.0.0/8". A host is allowed when its name matches, or when an address
// it resolves to is in a listed range. Listening is not limited.
func WithNetworkHosts(patterns ...string) NetworkOption {
	return func(c *networkConfig) { c.hosts = append(c.hosts, patterns...) }
}

// WithNetworkApprovalNeeded sets whether the named tools, or all of them when
// none are named, need approval. Later options win.
func WithNetworkApprovalNeeded(needed bool, tools ...string) NetworkOption {
	return func(c *networkConfig) {
		c.approval = append(c.approval, approvalRule{needed: needed, tools: tools})
	}
}

// NetworkToolset is the toolset returned by Network. Its connections,
// listeners and servers are shared by every agent that uses it.
type NetworkToolset struct {
	config networkConfig
	tools  *network.Tools
}

// Network returns a toolset for working with the network, the same on every
// OS:
//
//   - dns_lookup and whois query DNS servers and whois servers;
//   - http_get fetches public web pages; http_request sends any request;
//   - net_connect opens a TCP, UDP, TLS, unix socket or WebSocket connection,
//     net_send writes to it and net_read reads what arrived;
//   - net_listen accepts connections on a port, and http_serve runs a small
//     web server that answers with fixed responses and logs the requests;
//   - net_list and net_close show and close what is open.
//
// Lookups, http_get and handle management run without approval; tools that
// send data the model chose or open a port need approval (see
// WithNetworkApprovalNeeded). Anything read from the network goes into the
// model's context, so treat it as untrusted input.
//
// net_send, net_read and net_close calls from one model turn run one at a
// time in the order the model wrote them, so a read sees the reply to a send
// before it.
//
// Close the toolset to close everything it has open. The tools belong to the
// "network" toolset, so WithToolsets("network") gives an agent all of them.
func Network(opts ...NetworkOption) *NetworkToolset {
	config := networkConfig{private: true}
	for _, opt := range opts {
		opt(&config)
	}
	return &NetworkToolset{config: config, tools: network.New(config.private, config.hosts)}
}

// Register registers the network tools. It fails on an invalid host pattern
// or an unknown tool name in WithNetworkApprovalNeeded.
func (t *NetworkToolset) Register(registry ToolsRegistry) error {
	if err := t.tools.Init(); err != nil {
		return err
	}
	approval := maps.Clone(netApprovalDefaults)
	for _, rule := range t.config.approval {
		for _, name := range rule.tools {
			if _, ok := approval[name]; !ok {
				return fmt.Errorf("WithNetworkApprovalNeeded: unknown tool %q", name)
			}
		}
		for name := range approval {
			if len(rule.tools) == 0 || slices.Contains(rule.tools, name) {
				approval[name] = rule.needed
			}
		}
	}
	reg := func(name string) []ToolOption {
		opts := []ToolOption{WithToolset(ToolsetNetwork), WithApprovalNeeded(approval[name])}
		if netSequential[name] {
			opts = append(opts, WithSequential())
		}
		return opts
	}

	registerTextTool(registry, NetDNSLookup, "Look up DNS records, like dig. Asks the system's DNS server unless server is given. Output lists the answer, authority and additional sections with TTLs.", t.tools.DNSLookup, reg(NetDNSLookup)...)
	registerTextTool(registry, NetWhois, "Look up whois registration data for a domain, IP address or AS number. Starts at whois.iana.org and follows referrals to the authoritative server.", t.tools.Whois, reg(NetWhois)...)
	registerTextTool(registry, NetHTTPGet, "Fetch a URL with GET or HEAD. Only public internet addresses can be reached; use http_request for local or private hosts and for other methods.", t.tools.HTTPGet, reg(NetHTTPGet)...)
	registerTextTool(registry, NetHTTPRequest, "Send an HTTP request with any method, headers and body, to any allowed host including local ones.", t.tools.HTTPRequest, reg(NetHTTPRequest)...)
	registerTextTool(registry, NetConnect, "Open a connection and return its handle id. protocol is tcp, udp, tls, unix, unixgram, ws or wss. Incoming data is buffered until read with net_read; send with net_send.", t.tools.Connect, reg(NetConnect)...)
	registerTextTool(registry, NetSend, "Send data on a connection handle.", t.tools.Send, reg(NetSend)...)
	registerTextTool(registry, NetRead, "Read what has arrived on a handle, waiting up to timeout_ms for data. A connection returns its data; a listener returns newly accepted connections (each a new handle); an HTTP server returns the requests it received.", t.tools.Read, reg(NetRead)...)
	registerTextTool(registry, NetList, "List open handles: connections, listeners and HTTP servers.", t.tools.List, reg(NetList)...)
	registerTextTool(registry, NetClose, "Close a handle.", t.tools.CloseHandle, reg(NetClose)...)
	registerTextTool(registry, NetListen, "Listen for connections or datagrams and return the listener's handle id. A port alone, such as :8080, listens on 127.0.0.1; name 0.0.0.0:8080 to listen on every interface. Port 0 picks a free port.", t.tools.Listen, reg(NetListen)...)
	registerTextTool(registry, NetHTTPServe, "Start an HTTP server that answers with fixed responses and logs every request; read the log with net_read. Routes use Go ServeMux paths, such as /api/ or /items/{id}. A port alone listens on 127.0.0.1.", t.tools.HTTPServe, reg(NetHTTPServe)...)
	return nil
}

// Close closes every connection, listener and server the toolset has open.
// The toolset can still be used afterwards.
func (t *NetworkToolset) Close() error {
	return t.tools.Close()
}
