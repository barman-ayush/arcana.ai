import dotenv from "dotenv";
import { StreamingTextResponse } from "ai";
import { currentUser } from "@clerk/nextjs/server";
import { NextResponse } from "next/server";
import Replicate from "replicate";
import { MemoryManager } from "@/lib/memory";
import { ratelimit } from "@/lib/rate-limit";
import prismadb from "@/lib/prismadb";

const CONFIG = {
  TIMEOUT_MS: 30000,
  MAX_LENGTH: 512,
  MODELS: {
    default: "a16z-infra/llama-2-13b-chat:df7690f1994d94e96ad9d568eac121aecf50684a0b0963b25a41cc40061269e5"
  }
} as const;

// Pre-compile word analysis functions
const createWordMap = (words: string[]) => {
  const wordMap = new Map();
  words.forEach(word => {
    wordMap.set(word, (wordMap.get(word) || 0) + 1);
  });
  return wordMap;
};

const analyzeMessageSimilarity = (msg: string, prompt: string): boolean => {
  if (!msg) return false;

  const msgWords = msg.toLowerCase().split(" ").filter(word => word.length > 0);
  const promptWords = prompt.toLowerCase().split(" ").filter(word => word.length > 0);

  const msgWordMap = createWordMap(msgWords);
  const promptWordMap = createWordMap(promptWords);

  let commonWords = 0;
  msgWordMap.forEach((count, word) => {
    if (promptWordMap.has(word)) {
      commonWords += Math.min(count, promptWordMap.get(word) || 0);
    }
  });

  return commonWords / Math.max(msgWords.length, promptWords.length) > 0.6;
};

export async function POST(request: Request, { params }: { params: any}) {
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), CONFIG.TIMEOUT_MS);

  try {
    const chatId = (await params).chatId; 
    const { prompt } = await request.json();
    const user = await currentUser();

    if (!user?.firstName || !user?.id) {
      return new NextResponse("Unauthorized", { status: 401 });
    }

    // Parallel execution of initial checks
    const [rateLimitResult, companion, memoryManager] = await Promise.all([
      ratelimit(`${request.url}-${user.id}`),
      prismadb.companion.findUnique({
        where: { id: chatId },
        include: {
          messages: {
            where: { userId: user.id },
            orderBy: { createdAt: "desc" },
            take: 10  // Reduced from 50 to improve initial load
          }
        }
      }),
      MemoryManager.getInstance()
    ]);

    if (!rateLimitResult.success) {
      return new NextResponse("Rate limit exceeded", { status: 429 });
    }
    
    if (!companion) {
      return new NextResponse("Companion not found", { status: 404 });
    }

    // Create message in parallel with other operations
    const messagePromise = prismadb.message.create({
      data: {
        content: prompt,
        role: "user",
        userId: user.id,
        companionId: companion.id
      }
    });

    // Optimize message analysis
    const recentMessages = companion.messages || [];
    const isRepetitive = recentMessages.some(msg => 
      analyzeMessageSimilarity(msg.content || '', prompt)
    );
    const lastResponse = recentMessages[0]?.content || "";

    // Memory operations in parallel
    const companionKey = {
      companionName: companion.id,
      userId: user.id,
      modelName: "meta/meta-llama-3-8b-instruct"
    };

    const [records, similarDocs] = await Promise.all([
      memoryManager.readLatestHistory(companionKey),
      memoryManager.vectorSearch(companion.id, `${companion.id}.txt`)
    ]);

    // Conditional seeding only if necessary
    if (records.length === 0) {
      await memoryManager.seedChatHistory(companion.seed, "\n\n", companionKey);
    }

    // Format message history concisely
    const messageHistory = recentMessages
      .map(msg => `${msg.role === 'user' ? 'User' : companion.name}: ${msg.content}`)
      .reverse()
      .join("\n");

    const relevantHistory = similarDocs?.length
      ? similarDocs.map(doc => doc.pageContent).join("\n")
      : "";

    // Initialize Replicate early
    const replicate = new Replicate({
      auth: process.env.REPLICATE_API_TOKEN!
    });

    // Wait for message creation to complete
    await messagePromise;

    // Generate response with optimized prompt
    const modelResponse = await replicate.run(CONFIG.MODELS.default, {
      input: {
        prompt: `<|system|>
You are ${companion.name}. Focus on: ${prompt}
${companion.instructions}
Recent context:
${messageHistory}
Current question: ${prompt}
<|assistant|>`,
        temperature: 0.98,
        max_tokens: CONFIG.MAX_LENGTH,
        top_p: 0.95,
        presence_penalty: 1.8
      }
    });

    const finalResponse = String(modelResponse).split("\n")[0];

    if (finalResponse?.length > 1) {
      // Update history and create response message in parallel
      await Promise.all([
        memoryManager.writeToHistory(`User: ${prompt}\n${finalResponse.trim()}`, companionKey),
        prismadb.companion.update({
          where: { id: chatId },
          data: {
            messages: {
              create: {
                content: finalResponse.trim(),
                role: "system",
                userId: user.id
              }
            }
          }
        })
      ]);
    }

    return new StreamingTextResponse(
      new ReadableStream({
        async start(controller) {
          controller.enqueue(new TextEncoder().encode(finalResponse));
          controller.close();
        }
      })
    );
  } catch (error) {
    clearTimeout(timeoutId);
    console.error("[CHAT_POST]", error);
    return new NextResponse("Internal Error", { status: 500 });
  }
}