package project.door.lock.pack;

import project.door.lock.pack.R;
import android.app.Activity;
import android.os.Bundle;
import android.view.View;
import android.view.View.OnClickListener;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.Toast;

public class LightsActivity extends Activity implements OnClickListener
{
	
	Button btn_all,btn_one,btn_two,btn_three,btn_four;
	ImageView img_all,img_one,img_two,img_three,img_four;
	boolean all,one,two,three,four;
	public void onCreate(Bundle b)
	{
		super.onCreate(b);
		setContentView(R.layout.light);
		btn_all=(Button)findViewById(R.id.lights_btn_all);
		btn_all.setOnClickListener(this);
		
		btn_one=(Button)findViewById(R.id.lights_btn_one);
		btn_one.setOnClickListener(this);
		
		btn_two=(Button)findViewById(R.id.lights_btn_two);
		btn_two.setOnClickListener(this);
		
		btn_three=(Button)findViewById(R.id.lights_btn_three);
		btn_three.setOnClickListener(this);
		
		btn_four=(Button)findViewById(R.id.lights_btn_four);
		btn_four.setOnClickListener(this);
		
		img_all=(ImageView)findViewById(R.id.all_lights);
		img_one=(ImageView)findViewById(R.id.light_one);
		img_two=(ImageView)findViewById(R.id.light_two);
		img_three=(ImageView)findViewById(R.id.light_three);
		img_four=(ImageView)findViewById(R.id.light_four);
	}
	public void onClick(View v) {
		// TODO Auto-generated method stub
		if(v.equals(btn_all))
		{
			if(all)
			{
				all=false;
				MainActivity.getBluetoothChatObject().write("0".getBytes());
				btn_all.setText("All Lights Off");
				img_all.setImageResource(R.drawable.all_lights_on);
				Toast.makeText(this, "All Lights On", Toast.LENGTH_SHORT).show();
			}
			else
			{
				all=true;
				MainActivity.getBluetoothChatObject().write("9".getBytes());
				btn_all.setText("All Lights On");
				img_all.setImageResource(R.drawable.all_lights_off);
				Toast.makeText(this, "All Lights Off", Toast.LENGTH_SHORT).show();
				
			}
			
		}
		else if(v.equals(btn_one))
		{
			if(one)
			{
				one=false;
				MainActivity.getBluetoothChatObject().write("5".getBytes());
				btn_one.setText("Light One Off");
				img_one.setImageResource(R.drawable.on);
				Toast.makeText(this, "Light One On", Toast.LENGTH_SHORT).show();
			}
			else
			{
				one=true;
				MainActivity.getBluetoothChatObject().write("1".getBytes());
				btn_one.setText("Light One On");
				img_one.setImageResource(R.drawable.off);
				Toast.makeText(this, "Light One Off", Toast.LENGTH_SHORT).show();
			}
			
		}
		else if(v.equals(btn_two))
		{
			if(two)
			{
				one=false;
				MainActivity.getBluetoothChatObject().write("6".getBytes());
				btn_two.setText("Light Two Off");
				img_two.setImageResource(R.drawable.on);
				Toast.makeText(this, "Light Two On", Toast.LENGTH_SHORT).show();
				
			}
			else
			{
				two=true;
				MainActivity.getBluetoothChatObject().write("2".getBytes());
				btn_two.setText("Light Two On");
				img_two.setImageResource(R.drawable.off);
				Toast.makeText(this, "Light Two Off", Toast.LENGTH_SHORT).show();
			}
			
		}
		else if(v.equals(btn_three))
		{
			if(three)
			{
				three=false;
				MainActivity.getBluetoothChatObject().write("7".getBytes());
				btn_three.setText("Light Three Off");
				img_three.setImageResource(R.drawable.on);
				Toast.makeText(this, "Light Three On", Toast.LENGTH_SHORT).show();
			}
			else
			{
				three=true;
				MainActivity.getBluetoothChatObject().write("3".getBytes());
				btn_three.setText("Light Three On");
				img_three.setImageResource(R.drawable.off);
				Toast.makeText(this, "Light Three Off", Toast.LENGTH_SHORT).show();
			}
			
		}
		else if(v.equals(btn_four))
		{
			if(four)
			{
				four=false;
				MainActivity.getBluetoothChatObject().write("8".getBytes());
				btn_four.setText("Light Four Off");
				img_four.setImageResource(R.drawable.on);
				Toast.makeText(this, "Light Four On", Toast.LENGTH_SHORT).show();
				
			}
			else
			{
				four=true;
				MainActivity.getBluetoothChatObject().write("4".getBytes());
				btn_four.setText("Light Four On");
				img_four.setImageResource(R.drawable.off);
				Toast.makeText(this, "Light Four Off", Toast.LENGTH_SHORT).show();
			}
			
		}
		
	}

}
