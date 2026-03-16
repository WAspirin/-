"""
PDF 读取工具
"""

import PyPDF2
import sys

def read_pdf(pdf_path: str, max_pages: int = None):
    """读取 PDF 文件内容"""
    
    with open(pdf_path, 'rb') as f:
        reader = PyPDF2.PdfReader(f)
        
        print(f"PDF 文件：{pdf_path}")
        print(f"总页数：{len(reader.pages)}")
        print("="*60)
        
        n_pages = max_pages if max_pages else len(reader.pages)
        
        for i in range(min(n_pages, len(reader.pages))):
            print(f"\n--- 第 {i+1} 页 ---")
            page = reader.pages[i]
            text = page.extract_text()
            print(text)
        
        print("\n" + "="*60)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法：python read_pdf.py <pdf 文件路径> [最大页数]")
        sys.exit(1)
    
    pdf_path = sys.argv[1]
    max_pages = int(sys.argv[2]) if len(sys.argv) > 2 else None
    
    read_pdf(pdf_path, max_pages)
