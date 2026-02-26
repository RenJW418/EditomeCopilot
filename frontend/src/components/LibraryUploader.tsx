import { useState, useRef } from 'react';
import { Upload, CheckCircle, AlertCircle, Loader2 } from 'lucide-react';
import { cn } from '../lib/utils';

export function LibraryUploader() {
  const [isUploading, setIsUploading] = useState(false);
  const [status, setStatus] = useState<'idle' | 'success' | 'error'>('idle');
  const [message, setMessage] = useState('');
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFileChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    // Reset status
    setIsUploading(true);
    setStatus('idle');
    setMessage('');

    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await fetch('/api/upload_library', {
        method: 'POST',
        body: formData,
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.detail || 'Upload failed');
      }

      setStatus('success');
      setMessage(data.message || 'Library imported successfully!');
    } catch (err: any) {
      console.error(err);
      setStatus('error');
      setMessage(err.message || 'Failed to upload library.');
    } finally {
      setIsUploading(false);
      // Clear input so same file can be selected again if needed
      if (fileInputRef.current) fileInputRef.current.value = '';
    }
  };

  const handleClick = () => {
    fileInputRef.current?.click();
  };

  return (
    <div className="flex flex-col items-center gap-2">
      <input
        type="file"
        ref={fileInputRef}
        onChange={handleFileChange}
        accept=".bib,.ris,.txt"
        className="hidden"
      />
      
      <button
        onClick={handleClick}
        disabled={isUploading}
        className={cn(
          "flex items-center gap-2 px-3 py-1.5 text-xs font-medium rounded-md transition-colors",
          "bg-zinc-100 hover:bg-zinc-200 text-zinc-600 border border-zinc-200",
          isUploading && "opacity-75 cursor-wait"
        )}
        title="Import BibTeX (.bib) or EndNote (.ris) library"
      >
        {isUploading ? (
          <Loader2 className="w-3.5 h-3.5 animate-spin" />
        ) : (
          <Upload className="w-3.5 h-3.5" />
        )}
        Import Library (BibTeX/RIS)
      </button>

      {status !== 'idle' && (
        <div className={cn(
          "flex items-center gap-1.5 text-[10px] px-2 py-1 rounded-md animate-in fade-in slide-in-from-top-1",
          status === 'success' ? "bg-green-50 text-green-700" : "bg-red-50 text-red-700"
        )}>
          {status === 'success' ? <CheckCircle className="w-3 h-3" /> : <AlertCircle className="w-3 h-3" />}
          <span className="max-w-[150px] truncate">{message}</span>
        </div>
      )}
    </div>
  );
}
