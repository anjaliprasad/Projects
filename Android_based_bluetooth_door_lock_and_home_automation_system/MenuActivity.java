package project.door.lock.pack;

import android.app.Activity;
import android.content.Intent;
import android.os.Bundle;
import android.view.View;
import android.view.View.OnClickListener;
import android.widget.Button;
import android.widget.ImageView;

public class MenuActivity extends Activity implements OnClickListener
{
	ImageView btn_lock,btn_light;
	public void onCreate(Bundle b)
	{
		super.onCreate(b);
		setContentView(R.layout.menu_layout);
		  btn_lock=(ImageView)findViewById(R.id.menu_layout_btn_lock);
	      btn_lock.setOnClickListener(this);
	      
	      btn_light=(ImageView)findViewById(R.id.menu_layout_btn_light);
	      btn_light.setOnClickListener(this);
	}
	public void onClick(View v) {
		// TODO Auto-generated method stub

		if(v.equals(btn_lock))
		{
		   Intent it=new Intent(this,LockActivity.class);
		   startActivity(it);
		}
		else if(v.equals(btn_light))
		{
			 Intent it=new Intent(this,LightsActivity.class);
			 startActivity(it);
		}
	}

}
