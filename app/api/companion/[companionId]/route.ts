import { currentUser, auth } from "@clerk/nextjs/server";
import { NextResponse } from "next/server";

import prismadb from "@/lib/prismadb";

export async function PATCH(
  req: Request,
  { params }: { params: any }
) {
  try {
    const body = await req.json();
    const user = await currentUser();
    const { src, name, description, instructions, seed, categoryId } = body;

    console.log(name, src)

    if (!params.companionId) {
      return new NextResponse("Companion ID is Required.", { status: 400 });
    }

    if (!user || !user.id || !user.firstName) {
      return new NextResponse("Unauthorized", { status: 401 });
    }

    if (
      !src ||
      !name ||
      !description ||
      !instructions ||
      !seed ||
      !categoryId
    ) {
      return new NextResponse("Missing Required Field.", { status: 400 });
    }

    const companion = await prismadb.companion.update({
      where: {
        id: params.companionId,
        userId: user.id
      },
      data: {
        categoryId,
        userId: user.id,
        userName: user.firstName,
        src,
        name,
        description,
        instructions,
        seed
      }
    });

    return NextResponse.json(companion);
  } catch (error) {
    console.error("[COMPANION_PATCH]", error);
    return new NextResponse("Internal Error", { status: 500 });
  }
}

export async function DELETE(
  req: Request,
  { params }: { params: any }
) {
  try {
    const session = await auth();
    const userId = session?.userId;

    if (!userId) return new NextResponse("Unauthorized", { status: 401 });

    const companion = await prismadb.companion.delete({
      where: {
        userId,
        id: (await params).companionId
      }
    });

    return NextResponse.json(companion);
  } catch (error) {
    console.error("[COMPANION_DELETE]", error);
    return new NextResponse("Internal Error", { status: 500 });
  }
}
