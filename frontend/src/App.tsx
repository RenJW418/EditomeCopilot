import { useState, useRef, useEffect } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { cn } from './lib/utils';
import { Send, Menu, Bot, Circle, Brain, Upload, Plus, MessageSquare, Trash2 } from 'lucide-react';

interface Message {
  role: 'user' | 'assistant';
  content: string;
}

interface ChatSession {
  id: string;
  title: string;
  messages: Message[];
  timestamp: number;
}

export default function App() {
  const [sessions, setSessions] = useState<ChatSession[]>([]);
  const [currentSessionId, setCurrentSessionId] = useState<string | null>(null);
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [isUploading, setIsUploading] = useState(false);
  const scrollRef = useRef<HTMLDivElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Load sessions from API on mount
  useEffect(() => {
    fetchSessions();
  }, []);
  
  const fetchSessions = async () => {
    try {
      const res = await fetch('/api/sessions');
      if (res.ok) {
        const data = await res.json();
        // Backend returns lightweight sessions (id, title, timestamp)
        // We map them to ensure messages is at least [] if missing
        const formatted = data.map((s: any) => ({
             ...s,
             messages: [] // Content will be loaded on demand
        }));
        setSessions(formatted);
      }
    } catch (e) {
      console.error("Failed to fetch sessions", e);
    }
  };

  // Sync current messages to session state NOT localStorage anymore
  useEffect(() => {
    if (!currentSessionId) return;
    
    // We only update local state here for UI responsiveness.
    // Persistence handles via API calls during interaction.
    setSessions(prev => {
        return prev.map(s => 
          s.id === currentSessionId ? { ...s, messages } : s
        );
    });
  }, [messages, currentSessionId]);

  const handleNewChat = () => {
    setMessages([]);
    setCurrentSessionId(null);
    setInput('');
  };

  const handleLoadSession = async (sessionId: string) => {
    // If messages are already loaded, just switch
    const cached = sessions.find(s => s.id === sessionId);
    if (cached && cached.messages && cached.messages.length > 0) {
        setCurrentSessionId(sessionId);
        setMessages(cached.messages);
        return;
    }

    try {
        const res = await fetch(`/api/sessions/${sessionId}`);
        if (res.ok) {
            const data = await res.json();
            setCurrentSessionId(sessionId);
            setMessages(data.messages || []);
            
            // Update cache
            setSessions(prev => prev.map(s => s.id === sessionId ? { ...s, messages: data.messages } : s));
        }
    } catch (e) {
        console.error("Failed to load session", e);
    }
  };

  const handleDeleteSession = async (e: React.MouseEvent, sessionId: string) => {
    e.stopPropagation();
    try {
        await fetch(`/api/sessions/${sessionId}`, { method: 'DELETE' });
        const updated = sessions.filter(s => s.id !== sessionId);
        setSessions(updated);
        if (currentSessionId === sessionId) {
          handleNewChat();
        }
    } catch (err) {
        console.error("Delete failed", err);
    }
  };

  const handleLibraryUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    setIsUploading(true);
    // setUploadStatus('idle');
    // setUploadMessage('');

    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await fetch('/api/upload_library', {
        method: 'POST',
        body: formData,
      });

      const data = await response.json();
      if (!response.ok) throw new Error(data.detail || 'Upload failed');

      // setUploadStatus('success');
      // setUploadMessage(data.message);
      // Add a system message to chat
      setMessages(prev => [...prev, { 
        role: 'assistant', 
        content: `✅ **Library Imported**: ${data.message}\nNew citations are now available for retrieval.` 
      }]);
    } catch (err: any) {
      // setUploadStatus('error');
      // setUploadMessage(err.message);
      setMessages(prev => [...prev, { 
        role: 'assistant', 
        content: `❌ **Import Failed**: ${err.message}` 
      }]);
    } finally {
      setIsUploading(false);
      if (fileInputRef.current) fileInputRef.current.value = '';
    }
  };

  const handleSubmit = async (e?: React.FormEvent) => {
    e?.preventDefault();
    if (!input.trim() || loading) return;

    const userMsg = input;
    setInput('');
    
    // Capture current session ID before async operation
    let activeSessionId = currentSessionId;
    const isNew = !activeSessionId;

    // Optimistic UI update
    const userMessage: Message = { role: 'user', content: userMsg };
    
    if (isNew) {
        setMessages([userMessage]);
    } else {
        setMessages(prev => [...prev, userMessage]);
    }
    setLoading(true);

    try {
      const payload = {
          query: userMsg,
          history: isNew ? [] : messages,
          session_id: activeSessionId
      };

      const res = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });
      
      if (!res.ok) throw new Error('Network response was not ok');
      
      const data = await res.json();
      const assistantMsg: Message = { role: 'assistant', content: data.response };
      
      setMessages(prev => [...prev, assistantMsg]);
      
      // If it was a new session, the backend created an ID and Title for us
      if (isNew && data.session_id) {
          setCurrentSessionId(data.session_id);
          const newSession: ChatSession = {
              id: data.session_id,
              title: data.title,
              messages: [userMessage, assistantMsg],
              timestamp: Date.now()
          };
          setSessions(prev => [newSession, ...prev]);
      } else if (activeSessionId) {
          // Update existing session in the list to reflect new timestamp/preview ?
          // Or just leave it, since we fetch on load. 
          // But better UX to update the preview if needed.
      }
      
    } catch (err) {
      console.error(err);
      setMessages(prev => [...prev, { role: 'assistant', content: "Error: Could not connect to the backend server." }]);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    scrollRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, loading]);

  return (
    <div className="flex h-screen bg-white text-zinc-900 font-sans overflow-hidden">
      {/* Sidebar */}
      <div className="w-64 border-r border-zinc-200 flex-col hidden md:flex bg-zinc-50/50">
        {/* Sidebar Header */}
        <div className="p-4 border-b border-zinc-200/50 flex flex-col gap-4">
           <div className="flex items-center gap-2">
               <div className="w-8 h-8 rounded-full bg-zinc-200 flex items-center justify-center">
                 <span className="text-xs font-bold text-zinc-600">GE</span>
               </div>
               <span className="font-semibold text-sm">Gene Editing Almanac</span>
           </div>
           
           <button 
             onClick={handleNewChat}
             className="w-full flex items-center gap-2 px-3 py-2 text-sm font-medium text-zinc-700 bg-white border border-zinc-200 hover:bg-zinc-50 hover:border-zinc-300 rounded-md transition-all shadow-sm group"
           >
             <Plus className="w-4 h-4 text-zinc-500 group-hover:text-emerald-600 transition-colors" />
             New Conversation
           </button>
        </div>
        
        {/* Chat List */}
        <div className="flex-1 overflow-y-auto p-2 space-y-1">
           <div className="px-3 py-2 text-xs font-medium text-zinc-500 uppercase tracking-wider flex items-center justify-between">
             <span>Recent Chats</span>
             <span className="text-[10px] bg-zinc-100 px-1.5 py-0.5 rounded-full">{sessions.length}</span>
           </div>
           
           {sessions.length === 0 ? (
             <div className="px-3 py-4 text-xs text-zinc-400 italic text-center">
               No history yet. Start a new chat!
             </div>
           ) : (
             sessions.map(session => (
               <div 
                 key={session.id}
                 onClick={() => handleLoadSession(session.id)}
                 className={cn(
                   "group relative flex items-center justify-between w-full px-3 py-2.5 text-sm rounded-md transition-all cursor-pointer",
                   currentSessionId === session.id 
                     ? "bg-zinc-200/60 text-zinc-900 font-medium shadow-sm"
                     : "text-zinc-600 hover:bg-zinc-100/80 hover:text-zinc-900" 
                 )}
               >
                 <div className="flex items-center gap-2 overflow-hidden">
                   <MessageSquare className={cn("w-3.5 h-3.5 shrink-0 transition-colors", currentSessionId === session.id ? "text-emerald-600" : "text-zinc-400 group-hover:text-zinc-500")} />
                   <span className="truncate">{session.title || "Untitled Chat"}</span>
                 </div>
                 
                 <button
                   onClick={(e) => handleDeleteSession(e, session.id)}
                   className="opacity-0 group-hover:opacity-100 p-1 hover:bg-zinc-300/50 rounded transition-all absolute right-2"
                   title="Delete Chat"
                 >
                   <Trash2 className="w-3.5 h-3.5 text-zinc-500 hover:text-red-500" />
                 </button>
               </div>
             ))
           )}
        </div>
        
        <div className="p-4 border-t border-zinc-200">
           <div className="flex items-center gap-2 text-xs text-zinc-500">
             <div className="w-2 h-2 rounded-full bg-emerald-500"></div>
             System Operational
           </div>
        </div>
      </div>

      {/* Main Content */}
      <div className="flex-1 flex flex-col relative w-full">
        {/* Header */}
        <header className="h-14 border-b border-zinc-100 flex items-center px-4 justify-between bg-white z-10 shrink-0">
          <div className="flex items-center gap-2 text-sm text-zinc-500">
             <button className="md:hidden p-1 hover:bg-zinc-100 rounded">
               <Menu className="w-4 h-4" />
             </button>
             <span className="hidden sm:inline">Drafts</span>
             <span className="hidden sm:inline">/</span>
             <span className="font-medium text-zinc-900 flex items-center gap-2">
               Gene Editing Assistant
             </span>
          </div>
          <div className="text-xs font-medium text-emerald-600 bg-emerald-50 px-2 py-1 rounded-full flex items-center gap-1">
            <Circle className="w-2 h-2 fill-emerald-600" />
            Ready
          </div>
        </header>

        {/* Chat Area */}
        <div className="flex-1 overflow-y-auto px-4 sm:px-6 scroll-smooth">
           <div className="max-w-3xl mx-auto w-full py-8 pb-32 space-y-8">
             {messages.length === 0 && (
               <div className="pt-24 flex flex-col items-center justify-center text-center space-y-6 opacity-60">
                  <div className="w-16 h-16 bg-zinc-100 rounded-2xl flex items-center justify-center">
                     <Brain className="w-8 h-8 text-zinc-400" />
                  </div>
                  <div>
                    <h2 className="text-xl font-semibold text-zinc-800">Next-Gen RAG System</h2>
                    <p className="text-zinc-500 mt-2">Specialized in Gene Editing & Clinical Data</p>
                  </div>
               </div>
             )}
             
             {messages.map((msg, i) => (
               <div key={i} className={cn("flex w-full gap-4", msg.role === 'user' ? "justify-end" : "justify-start")}>
                 {msg.role === 'assistant' && (
                   <div className="hidden sm:flex w-8 h-8 rounded-full border border-zinc-200 items-center justify-center shrink-0 mt-1 bg-white">
                     <Bot className="w-4 h-4 text-zinc-500" />
                   </div>
                 )}
                 <div className={cn("max-w-[85%] sm:max-w-[75%]", 
                    msg.role === 'user' ? "ml-auto" : "mr-auto"
                 )}>
                    <div className={cn("px-5 py-3.5 rounded-2xl text-[15px] leading-relaxed shadow-sm",
                      msg.role === 'user' 
                        ? "bg-zinc-100 text-zinc-900 rounded-tr-sm" 
                        : "bg-white border border-zinc-100 text-zinc-800 shadow-zinc-200/50"
                    )}>
                      {msg.role === 'assistant' ? (
                        <div className="prose prose-sm prose-zinc max-w-none [&>p]:mb-2 [&>p:last-child]:mb-0">
                          <ReactMarkdown remarkPlugins={[remarkGfm]}>
                            {msg.content}
                          </ReactMarkdown>
                        </div>
                      ) : (
                        msg.content.split('\n').map((line: string, j: number) => (
                          <p key={j} className="mb-2 last:mb-0 min-h-[1.2em]">{line}</p>
                        ))
                      )}
                    </div>
                 </div>
               </div>
             ))}
             
             {loading && (
               <div className="flex w-full gap-4 justify-start pl-2">
                  <div className="hidden sm:flex w-8 h-8 rounded-full border border-zinc-200 items-center justify-center shrink-0 bg-white">
                     <Bot className="w-4 h-4 text-zinc-500" />
                   </div>
                   <div className="flex items-center gap-2 text-zinc-400 text-sm px-4 py-2">
                     <span className="animate-pulse">Thinking...</span>
                   </div>
               </div>
             )}
             <div ref={scrollRef} className="h-4" />
           </div>
        </div>

        {/* Input Area */}
        <div className="absolute bottom-0 left-0 right-0 p-4 sm:p-6 bg-gradient-to-t from-white via-white/80 to-transparent pointer-events-none">
           <div className="max-w-3xl mx-auto w-full pointer-events-auto">
             <div className="bg-white rounded-xl border border-zinc-200 shadow-xl shadow-zinc-200/30 overflow-hidden ring-1 ring-zinc-900/5 focus-within:ring-zinc-300 focus-within:border-zinc-300 transition-all transform hover:shadow-2xl hover:shadow-zinc-200/50">
               <form onSubmit={handleSubmit} className="flex flex-col">
                 <textarea 
                   value={input}
                   onChange={(e) => setInput(e.target.value)}
                   onKeyDown={(e) => {
                     if (e.key === 'Enter' && !e.shiftKey) {
                       e.preventDefault();
                       handleSubmit();
                     }
                   }}
                   disabled={loading}
                   placeholder="Ask about gene editing..."
                   className="w-full p-4 min-h-[60px] max-h-[200px] outline-none text-[16px] resize-none bg-transparent placeholder:text-zinc-400 disabled:opacity-50"
                   rows={1}
                 />
                 <div className="flex items-center justify-between px-3 py-2 bg-zinc-50/50 border-t border-zinc-100">
                    <div className="flex items-center gap-2">
                        {/* Hidden Upload Input */}
                        <input 
                          type="file" 
                          ref={fileInputRef} 
                          className="hidden" 
                          accept=".bib,.ris,.txt" 
                          onChange={handleLibraryUpload} 
                        />
                        
                        {/* Upload Button */}
                        <button
                          type="button"
                          disabled={isUploading || loading}
                          onClick={() => fileInputRef.current?.click()}
                          className={cn(
                            "flex items-center gap-2 px-2 py-1 rounded cursor-pointer hover:bg-zinc-100 transition-colors group",
                            isUploading && "animate-pulse"
                          )}
                          title="Import Zotero/EndNote Library (Export as RIS/BibTeX)"
                        >
                          <div className="w-5 h-5 rounded bg-white border border-zinc-200 flex items-center justify-center text-zinc-500 group-hover:border-zinc-300">
                             <Upload className="w-3 h-3" />
                          </div>
                          <span className="text-xs font-medium text-zinc-600 group-hover:text-zinc-900">
                            {isUploading ? "Importing..." : "Import RIS/BibTeX"}
                          </span>
                        </button>
                    </div>

                    <button 
                      type="submit"
                      disabled={!input.trim() || loading}
                      className="bg-black text-white rounded-lg px-4 py-1.5 text-xs font-semibold hover:bg-zinc-800 disabled:opacity-50 disabled:cursor-not-allowed transition-all flex items-center gap-2 shadow-sm"
                    >
                      Send <Send className="w-3 h-3" />
                    </button>
                 </div>
               </form>
             </div>
             <div className="text-center text-[11px] text-zinc-400 mt-3 font-medium tracking-wide">
               AI can make mistakes. Verify important information.
             </div>
           </div>
        </div>
      </div>
    </div>
  );
}
